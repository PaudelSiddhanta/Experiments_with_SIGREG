import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

from models.backbone import build_feature_encoder
from losses.sigreg import SIGReg


# ---------------------------------------------------------
# Basic components
# ---------------------------------------------------------

def mlp(
    input_dim,
    output_dim,
    hidden_dim,
):
    """
    BYOL-style two-layer MLP:

        Linear
        BatchNorm
        ReLU
        Linear
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, output_dim),
    )


def cosine_loss(x, y):
    """
    BYOL normalized cosine-distance loss.

    Returns one loss per sample.
    """
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)

    return 2.0 - 2.0 * (x * y).sum(dim=-1)


# ---------------------------------------------------------
# Augmentation
# ---------------------------------------------------------

def build_byol_augmentation(image_size):
    """
    Closely matches the default augmentation currently used
    by byol-pytorch.

    Input is already a tensor from the CIFAR DataLoader.
    """

    return nn.Sequential(
        T.RandomApply(
            [
                T.ColorJitter(
                    brightness=0.8,
                    contrast=0.8,
                    saturation=0.8,
                    hue=0.2,
                )
            ],
            p=0.3,
        ),

        T.RandomGrayscale(p=0.2),

        T.RandomHorizontalFlip(),

        T.RandomApply(
            [
                T.GaussianBlur(
                    kernel_size=(3, 3),
                    sigma=(1.0, 2.0),
                )
            ],
            p=0.2,
        ),

        T.RandomResizedCrop(
            size=(image_size, image_size)
        ),

        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    )


# ---------------------------------------------------------
# EMA
# ---------------------------------------------------------

@torch.no_grad()
def ema_update(
    online_module,
    target_module,
    decay,
):
    """
    target <- decay * target
              + (1-decay) * online
    """

    for online_param, target_param in zip(
        online_module.parameters(),
        target_module.parameters(),
    ):
        target_param.data.mul_(decay)
        target_param.data.add_(
            online_param.data,
            alpha=1.0 - decay,
        )


def freeze_module(module):
    for parameter in module.parameters():
        parameter.requires_grad = False


# ---------------------------------------------------------
# Main configurable SSL model
# ---------------------------------------------------------

class SSLModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        method = config["method"]

        self.use_ema = method["use_ema"]
        self.stop_gradient = method["stop_gradient"]
        self.use_predictor = method["use_predictor"]
        self.use_sigreg = method["use_sigreg"]

        self.ema_decay = config["byol"][
            "moving_average_decay"
        ]

        projection_size = config["byol"][
            "projection_size"
        ]

        projection_hidden_size = config["byol"][
            "projection_hidden_size"
        ]

        # -------------------------------------------------
        # Online/shared encoder
        # -------------------------------------------------

        self.encoder = build_feature_encoder(config)

        self.projector = mlp(
            input_dim=512,
            output_dim=projection_size,
            hidden_dim=projection_hidden_size,
        )

        # -------------------------------------------------
        # Optional predictor
        # -------------------------------------------------

        if self.use_predictor:
            self.predictor = mlp(
                input_dim=projection_size,
                output_dim=projection_size,
                hidden_dim=projection_hidden_size,
            )
        else:
            self.predictor = None

        # -------------------------------------------------
        # EMA target network
        # -------------------------------------------------

        if self.use_ema:

            self.target_encoder = copy.deepcopy(
                self.encoder
            )

            self.target_projector = copy.deepcopy(
                self.projector
            )

            freeze_module(self.target_encoder)
            freeze_module(self.target_projector)

        else:
            self.target_encoder = None
            self.target_projector = None

        # -------------------------------------------------
        # SIGReg
        # -------------------------------------------------

        if self.use_sigreg:

            sigreg_config = config["sigreg"]

            self.sigreg = SIGReg(
                knots=sigreg_config["knots"],
                num_slices=sigreg_config[
                    "num_slices"
                ],
            )

            self.sigreg_weight = sigreg_config[
                "weight"
            ]

        else:
            self.sigreg = None
            self.sigreg_weight = 0.0

        # -------------------------------------------------
        # Augmentations
        # -------------------------------------------------

        image_size = config["data"]["image_size"]

        self.augment = build_byol_augmentation(
            image_size
        )

        self._validate_configuration()

    # -----------------------------------------------------
    # Configuration checks
    # -----------------------------------------------------

    def _validate_configuration(self):

        if self.use_ema and not self.stop_gradient:
            raise ValueError(
                "EMA target should use stop-gradient "
                "for the BYOL-style experiment."
            )

        if self.use_ema and not self.use_predictor:
            raise ValueError(
                "Our EMA BYOL experiments keep "
                "the predictor enabled."
            )

    # -----------------------------------------------------
    # Online/shared network
    # -----------------------------------------------------

    def online_projection(self, x):

        representation = self.encoder(x)

        projection = self.projector(
            representation
        )

        return projection

    # -----------------------------------------------------
    # Target network
    # -----------------------------------------------------

    @torch.no_grad()
    def target_projection(self, x):

        representation = self.target_encoder(x)

        projection = self.target_projector(
            representation
        )

        return projection

    # -----------------------------------------------------
    # EMA update
    # -----------------------------------------------------

    @torch.no_grad()
    def update_target(self):

        if not self.use_ema:
            return

        ema_update(
            self.encoder,
            self.target_encoder,
            self.ema_decay,
        )

        ema_update(
            self.projector,
            self.target_projector,
            self.ema_decay,
        )

    # -----------------------------------------------------
    # Forward
    # -----------------------------------------------------

    def forward(self, images):

        # Two independent augmentations of SAME batch.
        view1 = self.augment(images)
        view2 = self.augment(images)

        # Shared / online representations.
        z1 = self.online_projection(view1)
        z2 = self.online_projection(view2)

        # Predictor, if this variant uses one.
        if self.use_predictor:
            p1 = self.predictor(z1)
            p2 = self.predictor(z2)
        else:
            p1 = z1
            p2 = z2

        # =================================================
        # EMA TARGET CASE: A and B
        # =================================================

        if self.use_ema:

            with torch.no_grad():

                target_z1 = self.target_projection(
                    view1
                )

                target_z2 = self.target_projection(
                    view2
                )

            invariance_loss = (
                cosine_loss(
                    p1,
                    target_z2.detach(),
                ).mean()
                +
                cosine_loss(
                    p2,
                    target_z1.detach(),
                ).mean()
            )

        # =================================================
        # SHARED WEIGHTS CASE: C-F
        # =================================================

        else:

            if self.stop_gradient:

                # C:
                #
                # p1 learns to predict a detached z2
                # p2 learns to predict a detached z1

                target_z2 = z2.detach()
                target_z1 = z1.detach()

            else:

                # D/E/F:
                #
                # gradients flow through both sides

                target_z2 = z2
                target_z1 = z1

            invariance_loss = (
                cosine_loss(
                    p1,
                    target_z2,
                ).mean()
                +
                cosine_loss(
                    p2,
                    target_z1,
                ).mean()
            )

        # =================================================
        # SIGREG
        # =================================================

        if self.use_sigreg:

            # Use ONLINE/shared projections.
            #
            # Shape:
            #   [2 * batch_size, projection_dim]

            all_z = torch.cat(
                [z1, z2],
                dim=0,
            )

            sigreg_loss = self.sigreg(all_z)

        else:

            sigreg_loss = torch.zeros(
                (),
                device=images.device,
            )

        total_loss = (
            invariance_loss
            +
            self.sigreg_weight * sigreg_loss
        )

        return {
            "loss": total_loss,
            "invariance_loss": invariance_loss,
            "sigreg_loss": sigreg_loss,
        }

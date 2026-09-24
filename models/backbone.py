import torch.nn as nn
from torchvision import models

def build_backbone(config):
    name = config["model"]["backbone"]

    if name != "resnet18":
        raise ValueError(f"Unsupported backbone: {name}")

    model = models.resnet18(weights=None)

    if config["model"].get("cifar_stem", False):
        model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = nn.Identity()

    return model


def build_feature_encoder(config):
    """
    ResNet-18 returning the 512-dimensional representation
    after global average pooling.
    """
    model = build_backbone(config)
    model.fc = nn.Identity()

    return model

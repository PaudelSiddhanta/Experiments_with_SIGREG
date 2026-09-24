import os
import csv
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.ssl_model import SSLModel
from utils.representation_metrics import (
    compute_representation_metrics,
    alignment,
)


EXPERIMENTS = [
    "A_byol",
]

OUTPUT_ROOT = "/scratch/sp7007/byol_sigreg/outputs"
DATA_ROOT = "/scratch/sp7007/byol_sigreg/data"

BATCH_SIZE = 256
NUM_WORKERS = 8


def get_test_loader(normalize=True):

    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
    else:
        transform = transforms.ToTensor()

    dataset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=False,
        download=False,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return loader


@torch.no_grad()
def collect_single_view_representations(
    model,
    metric_loader,
    device,
):
    encoder_features = []
    projector_features = []
    predictor_features = []

    model.eval()

    for images, _ in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        h = model.encoder(images)
        z = model.projector(h)

        encoder_features.append(
            h.cpu()
        )

        projector_features.append(
            z.cpu()
        )

        if model.predictor is not None:
            p = model.predictor(z)

            predictor_features.append(
                p.cpu()
            )

    results = {
        "encoder": torch.cat(
            encoder_features,
            dim=0,
        ),

        "projector": torch.cat(
            projector_features,
            dim=0,
        ),
    }

    if len(predictor_features) > 0:
        results["predictor"] = torch.cat(
            predictor_features,
            dim=0,
        )

    return results


@torch.no_grad()
def collect_alignment_representations(
    model,
    alignment_loader,
    device,
):
    """
    Make two independently augmented views using the
    SAME augmentation pipeline used by the SSL model.

    Collect h, z, and p for both views.
    """

    first = {
        "encoder": [],
        "projector": [],
        "predictor": [],
    }

    second = {
        "encoder": [],
        "projector": [],
        "predictor": [],
    }

    model.eval()

    for images, _ in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        x1 = model.augment(images)
        x2 = model.augment(images)

        h1 = model.encoder(x1)
        h2 = model.encoder(x2)

        z1 = model.projector(h1)
        z2 = model.projector(h2)

        first["encoder"].append(
            h1.cpu()
        )

        second["encoder"].append(
            h2.cpu()
        )

        first["projector"].append(
            z1.cpu()
        )

        second["projector"].append(
            z2.cpu()
        )

        if model.predictor is not None:
            p1 = model.predictor(z1)
            p2 = model.predictor(z2)

            first["predictor"].append(
                p1.cpu()
            )

            second["predictor"].append(
                p2.cpu()
            )

    results = {}

    for level in [
        "encoder",
        "projector",
        "predictor",
    ]:
        if len(first[level]) == 0:
            continue

        results[level] = (
            torch.cat(
                first[level],
                dim=0,
            ),
            torch.cat(
                second[level],
                dim=0,
            ),
        )

    return results


def load_experiment(
    experiment,
    device,
):
    checkpoint_path = os.path.join(
        OUTPUT_ROOT,
        experiment,
        "seed_0",
        "checkpoints",
        "latest.pt",
    )

    print(
        f"\nLoading {experiment}"
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    config = checkpoint["config"]

    model = SSLModel(
        config
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model = model.to(device)

    model.eval()

    return model


def save_results(rows):
    output_path = os.path.join(
        OUTPUT_ROOT,
        "representation_metrics.csv",
    )

    all_fields = set()

    for row in rows:
        all_fields.update(
            row.keys()
        )

    preferred = [
        "experiment",
        "level",
        "num_samples",
        "dimension",
        "alignment",
        "total_feature_variance",
        "constant_collapse",
        "mean_feature_std",
        "min_feature_std",
        "near_zero_std_ratio",
        "mean_feature_norm",
        "rank_99",
        "effective_rank",
        "stable_rank",
        "covariance_offdiag_ratio",
        "mean_pairwise_cosine",
        "feature_entropy",
        "gini_sparsity",
    ]

    remaining = sorted(
        all_fields - set(preferred)
    )

    fieldnames = (
        preferred + remaining
    )

    with open(
        output_path,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(
        f"\nSaved metrics to:\n{output_path}"
    )


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    metric_loader = get_test_loader(normalize=True)
    alignment_loader = get_test_loader(normalize=False)

    rows = []

    for experiment in EXPERIMENTS:

        model = load_experiment(
            experiment,
            device,
        )

        features = (
            collect_single_view_representations(
                model,
                metric_loader,
                device,
            )
        )

        alignment_features = (
            collect_alignment_representations(
                model,
                alignment_loader,
                device,
            )
        )

        for level, x in features.items():

            print(
                f"\n{experiment} - {level}"
            )

            print(
                "shape:",
                tuple(x.shape),
            )

            metrics = (
                compute_representation_metrics(
                    x
                )
            )

            row = {
                "experiment":
                    experiment,

                "level":
                    level,

                "num_samples":
                    x.shape[0],

                "dimension":
                    x.shape[1],
            }

            if level in alignment_features:

                x1, x2 = (
                    alignment_features[level]
                )

                row["alignment"] = alignment(
                    x1,
                    x2,
                )

            row.update(metrics)

            rows.append(row)

            print(
                "rank_99:",
                metrics["rank_99"],
            )

            print(
                "effective_rank:",
                metrics["effective_rank"],
            )

            print(
                "stable_rank:",
                metrics["stable_rank"],
            )

            print(
                "mean_feature_std:",
                metrics["mean_feature_std"],
            )

            print(
                "mean_pairwise_cosine:",
                metrics["mean_pairwise_cosine"],
            )

            print(
                "alignment:",
                row.get(
                    "alignment",
                    None,
                ),
            )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_results(rows)


if __name__ == "__main__":
    main()

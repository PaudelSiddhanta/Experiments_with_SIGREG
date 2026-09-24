import argparse
import os

import torch
import torch.nn as nn
from tqdm import tqdm

from datasets.cifar10 import (
    build_linear_probe_loaders,
)
from models.backbone import build_feature_encoder
from utils.config import load_config
from utils.seed import set_seed
from utils.logging import CSVLogger


def evaluate(
    encoder,
    classifier,
    loader,
    device,
):
    encoder.eval()
    classifier.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            features = encoder(images)

            logits = classifier(features)

            predictions = logits.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    args = parser.parse_args()

    config = load_config(args.config)

    seed = config["experiment"]["seed"]

    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
    )

    # Build exact same architecture used for SSL.

    encoder = build_feature_encoder(config)

    encoder.load_state_dict(
     checkpoint["encoder"]
    )

    encoder = encoder.to(device)
    # Freeze everything.
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    encoder.eval()

    # ResNet-18 avgpool representation is 512-D.
    classifier = nn.Linear(
        512,
        10,
    ).to(device)

    train_loader, test_loader = (
        build_linear_probe_loaders(config)
    )

    criterion = nn.CrossEntropyLoss()

    probe_config = config["linear_probe"]

    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=probe_config["lr"],
        momentum=probe_config["momentum"],
        weight_decay=probe_config["weight_decay"],
    )

    run_dir = os.path.dirname(
        os.path.dirname(args.checkpoint)
    )

    logger = CSVLogger(
        os.path.join(
            run_dir,
            "linear_probe.csv",
        ),
        [
            "epoch",
            "train_loss",
            "test_accuracy",
        ],
    )

    epochs = probe_config["epochs"]

    best_accuracy = 0.0

    for epoch in range(1, epochs + 1):

        classifier.train()

        running_loss = 0.0
        num_batches = 0

        progress = tqdm(
            train_loader,
            desc=f"Probe {epoch}/{epochs}",
        )

        for images, labels in progress:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            # Absolutely no gradients through encoder.
            with torch.no_grad():
                features = encoder(images)

            logits = classifier(features)

            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        average_loss = (
            running_loss / num_batches
        )

        test_accuracy = evaluate(
            encoder,
            classifier,
            test_loader,
            device,
        )

        best_accuracy = max(
            best_accuracy,
            test_accuracy,
        )

        print(
            f"Epoch {epoch}: "
            f"loss={average_loss:.4f}, "
            f"test_acc={test_accuracy:.2f}%"
        )

        logger.log({
            "epoch": epoch,
            "train_loss": average_loss,
            "test_accuracy": test_accuracy,
        })

    print(
        f"\nBest linear probe accuracy: "
        f"{best_accuracy:.2f}%"
    )


if __name__ == "__main__":
    main()

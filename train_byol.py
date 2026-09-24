import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from byol_pytorch import BYOL
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")

    return parser.parse_args()


def build_encoder():
    """
    ResNet-18 adapted slightly for CIFAR-10's 32x32 images.
    """

    net = models.resnet18(weights=None)

    # Standard ResNet is designed for 224x224 ImageNet images.
    # CIFAR-10 is only 32x32, so use a smaller first convolution
    # and remove the initial max-pooling operation.
    net.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    net.maxpool = nn.Identity()

    return net


def main():
    args = get_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    os.makedirs(args.save_dir, exist_ok=True)

    # BYOL itself will create the two random augmented views.
    # Therefore the dataset transform only converts the image to a tensor.
    transform = transforms.ToTensor()

    dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    encoder = build_encoder()

    learner = BYOL(
        encoder,
        image_size=32,
        hidden_layer="avgpool",
    )

    learner = learner.to(device)

    optimizer = torch.optim.Adam(
        learner.parameters(),
        lr=args.lr,
    )

    for epoch in range(1, args.epochs + 1):

        learner.train()

        running_loss = 0.0

        progress = tqdm(
            loader,
            desc=f"Epoch {epoch}/{args.epochs}",
        )

        for images, _ in progress:

            images = images.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad()

            loss = learner(images)

            loss.backward()

            optimizer.step()

            # Update BYOL target network using EMA
            learner.update_moving_average()

            running_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        avg_loss = running_loss / len(loader)

        print(
            f"Epoch {epoch}: "
            f"average BYOL loss = {avg_loss:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "learner": learner.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": avg_loss,
        }

        torch.save(
            checkpoint,
            os.path.join(
                args.save_dir,
                f"byol_epoch_{epoch}.pt",
            ),
        )


if __name__ == "__main__":
    main()

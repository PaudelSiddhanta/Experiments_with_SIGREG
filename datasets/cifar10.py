import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.seed import seed_worker


# Match normalization currently used by byol-pytorch's default augmentation.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_ssl_loader(config):
    """
    BYOL performs its own SSL augmentations internally.

    Therefore here we only convert PIL images to tensors.
    """

    root = config["data"]["root"]
    batch_size = config["train"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    seed = config["experiment"]["seed"]

    dataset = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    return loader


def build_linear_probe_loaders(config):
    root = config["data"]["root"]
    batch_size = config["linear_probe"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    seed = config["experiment"]["seed"]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_dataset = datasets.CIFAR10(
        root=root,
        train=True,
        download=False,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=root,
        train=False,
        download=False,
        transform=test_transform,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader

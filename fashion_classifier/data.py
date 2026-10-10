"""Dataset loading: train / validation / test splits for Fashion-MNIST."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

# Per-channel statistics of the Fashion-MNIST training set
# (not the MNIST digits values, which are often copied by mistake).
MEAN = (0.2860,)
STD = (0.3530,)


def build_transform(augment: bool = False) -> transforms.Compose:
    """Tensor conversion + normalization, with optional light augmentation."""
    steps = []
    if augment:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28, padding=2),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


def get_test_set(data_dir: str = "data") -> datasets.FashionMNIST:
    return datasets.FashionMNIST(
        data_dir, train=False, download=True, transform=build_transform(False)
    )


def get_loaders(
    data_dir: str = "data",
    batch_size: int = 128,
    val_fraction: float = 0.1,
    augment: bool = False,
    seed: int = 0,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, validation, test) loaders.

    The validation split is carved out of the official training set with a
    fixed seed, so model selection never touches the test set. Validation
    images are never augmented.
    """
    train_full = datasets.FashionMNIST(
        data_dir, train=True, download=True, transform=build_transform(augment)
    )
    eval_full = datasets.FashionMNIST(
        data_dir, train=True, download=True, transform=build_transform(False)
    )

    n_val = int(len(train_full) * val_fraction)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train_full), generator=generator).tolist()
    val_idx, train_idx = order[:n_val], order[n_val:]

    common = dict(batch_size=batch_size, num_workers=num_workers)
    train_loader = DataLoader(Subset(train_full, train_idx), shuffle=True, **common)
    val_loader = DataLoader(Subset(eval_full, val_idx), shuffle=False, **common)
    test_loader = DataLoader(get_test_set(data_dir), shuffle=False, **common)
    return train_loader, val_loader, test_loader

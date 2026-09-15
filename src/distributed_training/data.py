"""Deterministic synthetic and MNIST data utilities."""

from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from distributed_training.config import TrainingConfig


def partition_indices(length: int, rank: int, world_size: int, *, equal: bool) -> list[int]:
    """Return a deterministic, non-overlapping strided partition.

    Training partitions can be forced to equal length so synchronous workers
    execute the same number of optimization steps. At most ``world_size - 1``
    trailing samples are then omitted.
    """

    if length < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid partition arguments")
    usable = length - (length % world_size) if equal else length
    return list(range(rank, usable, world_size))


def _synthetic_digits(size: int, prototype_seed: int, sample_seed: int) -> TensorDataset:
    prototype_generator = torch.Generator().manual_seed(prototype_seed)
    sample_generator = torch.Generator().manual_seed(sample_seed)
    prototypes = torch.randn(10, 1, 28, 28, generator=prototype_generator)
    prototypes = prototypes / prototypes.flatten(1).std(dim=1).view(-1, 1, 1, 1)
    labels = torch.arange(size, dtype=torch.long) % 10
    labels = labels[torch.randperm(size, generator=sample_generator)]
    noise = 0.30 * torch.randn(size, 1, 28, 28, generator=sample_generator)
    images = prototypes[labels] + noise
    images = (images - images.mean()) / images.std()
    return TensorDataset(images, labels)


def build_datasets(config: TrainingConfig) -> tuple[Dataset, Dataset]:
    """Build the requested datasets without importing torchvision unnecessarily."""

    config.validate()
    if config.dataset == "synthetic":
        return (
            _synthetic_digits(config.train_size, config.seed, config.seed + 101),
            _synthetic_digits(config.test_size, config.seed, config.seed + 202),
        )

    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    if config.train_size < len(train):
        train = Subset(train, range(config.train_size))
    if config.test_size < len(test):
        test = Subset(test, range(config.test_size))
    return train, test


def make_loader(
    dataset: Dataset,
    indices: Sequence[int],
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    loader_workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=loader_workers,
        generator=generator,
    )

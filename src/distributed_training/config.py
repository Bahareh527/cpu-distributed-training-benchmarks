"""Validated benchmark configuration."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Settings shared by every training architecture."""

    epochs: int = 3
    global_batch_size: int = 128
    learning_rate: float = 0.01
    dataset: str = "synthetic"
    train_size: int = 2048
    test_size: int = 512
    seed: int = 7
    loader_workers: int = 0
    threads_per_worker: int = 1

    def validate(self, world_size: int = 1) -> "TrainingConfig":
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if world_size < 1:
            raise ValueError("world_size must be at least 1")
        if self.global_batch_size < world_size:
            raise ValueError("global_batch_size must be at least the worker count")
        if self.global_batch_size % world_size:
            raise ValueError("global_batch_size must be divisible by the worker count")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.dataset not in {"synthetic", "mnist"}:
            raise ValueError("dataset must be 'synthetic' or 'mnist'")
        if self.train_size < world_size:
            raise ValueError("train_size must be at least the worker count")
        if self.test_size < world_size:
            raise ValueError("test_size must be at least the worker count")
        if self.loader_workers < 0:
            raise ValueError("loader_workers cannot be negative")
        if self.threads_per_worker < 1:
            raise ValueError("threads_per_worker must be at least 1")
        return self

    def local_batch_size(self, world_size: int) -> int:
        self.validate(world_size)
        return self.global_batch_size // world_size

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)

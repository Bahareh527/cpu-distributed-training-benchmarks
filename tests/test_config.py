import pytest

from distributed_training.config import TrainingConfig


def test_local_batch_size_preserves_global_batch() -> None:
    config = TrainingConfig(global_batch_size=128)
    assert config.local_batch_size(4) == 32


def test_global_batch_must_be_divisible_by_workers() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TrainingConfig(global_batch_size=127).validate(4)

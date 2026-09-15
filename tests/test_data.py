from distributed_training.config import TrainingConfig
from distributed_training.data import build_datasets, partition_indices


def test_equal_partitions_are_disjoint_and_balanced() -> None:
    partitions = [partition_indices(11, rank, 3, equal=True) for rank in range(3)]
    assert all(len(partition) == 3 for partition in partitions)
    assert len(set().union(*map(set, partitions))) == 9
    assert all(
        set(left).isdisjoint(right) for left, right in zip(partitions, partitions[1:], strict=False)
    )


def test_synthetic_dataset_is_reproducible() -> None:
    config = TrainingConfig(train_size=20, test_size=10)
    first_train, _ = build_datasets(config)
    second_train, _ = build_datasets(config)
    assert first_train[0][1].item() == second_train[0][1].item()
    assert first_train[0][0].equal(second_train[0][0])

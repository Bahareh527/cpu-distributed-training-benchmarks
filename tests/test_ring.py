import pytest

from distributed_training.ring import padded_chunk_size


@pytest.mark.parametrize(("elements", "workers", "expected"), [(8, 4, 2), (9, 4, 3), (1, 4, 1)])
def test_padded_chunk_size(elements: int, workers: int, expected: int) -> None:
    assert padded_chunk_size(elements, workers) == expected

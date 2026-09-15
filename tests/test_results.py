import pytest

from distributed_training.results import ResultRow, ResultWriter, speedup


def test_result_writer_uses_stable_schema(tmp_path) -> None:
    path = tmp_path / "result.csv"
    with ResultWriter(path) as writer:
        writer.write(ResultRow("single", 1, 1, 2.0, 0.5, 95.0))
    assert path.read_text(encoding="utf-8").splitlines()[0] == (
        "method,workers,epoch,epoch_time_sec,train_loss,test_accuracy"
    )


def test_speedup() -> None:
    assert speedup(10.0, 4.0) == 2.5
    with pytest.raises(ValueError):
        speedup(10.0, 0.0)

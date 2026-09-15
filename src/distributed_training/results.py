"""Stable CSV schema and benchmark aggregation."""

import csv
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResultRow:
    method: str
    workers: int
    epoch: int
    epoch_time_sec: float
    train_loss: float
    test_accuracy: float


FIELDS = list(ResultRow.__dataclass_fields__)


class ResultWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stream, fieldnames=FIELDS)
        self._writer.writeheader()

    def write(self, row: ResultRow) -> None:
        self._writer.writerow(asdict(row))
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> "ResultWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def speedup(single_time: float, distributed_time: float) -> float:
    if single_time <= 0 or distributed_time <= 0:
        raise ValueError("timings must be positive")
    return single_time / distributed_time

"""Generate comparable learning and scaling plots from benchmark CSV files."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from distributed_training.results import ResultRow, speedup


def load_results(directory: str | Path) -> list[ResultRow]:
    rows: list[ResultRow] = []
    for path in sorted(Path(directory).glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            for raw in csv.DictReader(stream):
                rows.append(
                    ResultRow(
                        method=raw["method"],
                        workers=int(raw["workers"]),
                        epoch=int(raw["epoch"]),
                        epoch_time_sec=float(raw["epoch_time_sec"]),
                        train_loss=float(raw["train_loss"]),
                        test_accuracy=float(raw["test_accuracy"]),
                    )
                )
    if not rows:
        raise ValueError(f"no benchmark CSV files found in {directory}")
    return rows


def _groups(rows: list[ResultRow]) -> dict[tuple[str, int], list[ResultRow]]:
    grouped: dict[tuple[str, int], list[ResultRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.method, row.workers)].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: row.epoch)
    return dict(grouped)


def _learning_curve(
    groups: dict[tuple[str, int], list[ResultRow]],
    output: Path,
    attribute: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(8, 5))
    for (method, workers), values in sorted(groups.items()):
        plt.plot(
            [row.epoch for row in values],
            [getattr(row, attribute) for row in values],
            marker="o",
            label=f"{method} ({workers})",
        )
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def create_plots(log_directory: str | Path, output_directory: str | Path) -> list[Path]:
    rows = load_results(log_directory)
    groups = _groups(rows)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    accuracy_path = output / "accuracy.png"
    loss_path = output / "loss.png"
    time_path = output / "epoch-time.png"
    speedup_path = output / "speedup.png"
    _learning_curve(groups, accuracy_path, "test_accuracy", "Test accuracy (%)")
    _learning_curve(groups, loss_path, "train_loss", "Training loss")

    labels = [f"{method}\n{workers} worker(s)" for method, workers in sorted(groups)]
    mean_times = [
        sum(row.epoch_time_sec for row in groups[key]) / len(groups[key]) for key in sorted(groups)
    ]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, mean_times)
    plt.ylabel("Mean epoch time (s)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(time_path, dpi=160)
    plt.close()

    single_keys = [key for key in groups if key == ("single", 1)]
    if not single_keys:
        raise ValueError("speedup requires a single-process baseline")
    baseline = sum(row.epoch_time_sec for row in groups[single_keys[0]]) / len(
        groups[single_keys[0]]
    )
    plt.figure(figsize=(8, 5))
    for method in sorted({key[0] for key in groups if key[0] != "single"}):
        worker_counts = sorted(key[1] for key in groups if key[0] == method)
        values = []
        for workers in worker_counts:
            method_rows = groups[(method, workers)]
            mean_time = sum(row.epoch_time_sec for row in method_rows) / len(method_rows)
            values.append(speedup(baseline, mean_time))
        plt.plot(worker_counts, values, marker="o", label=method)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1, label="baseline")
    plt.xlabel("Training workers")
    plt.ylabel("Speedup (single time / distributed time)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(speedup_path, dpi=160)
    plt.close()
    return [accuracy_path, loss_path, time_path, speedup_path]

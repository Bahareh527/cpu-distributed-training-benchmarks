"""Single-process reference implementation."""

from pathlib import Path
from time import perf_counter

import torch

from distributed_training.common import configure_runtime, evaluate_counts, train_one_epoch
from distributed_training.config import TrainingConfig
from distributed_training.data import build_datasets, make_loader
from distributed_training.model import DigitCNN
from distributed_training.results import ResultRow, ResultWriter


def run_single(config: TrainingConfig, output: str | Path) -> None:
    config.validate()
    configure_runtime(config.seed, config.threads_per_worker)
    train_data, test_data = build_datasets(config)
    train_loader = make_loader(
        train_data,
        range(len(train_data)),
        config.global_batch_size,
        shuffle=True,
        seed=config.seed,
        loader_workers=config.loader_workers,
    )
    test_loader = make_loader(
        test_data,
        range(len(test_data)),
        min(512, len(test_data)),
        shuffle=False,
        seed=config.seed,
        loader_workers=config.loader_workers,
    )
    model = DigitCNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    with ResultWriter(output) as writer:
        for epoch in range(1, config.epochs + 1):
            started = perf_counter()
            loss_sum, sample_count = train_one_epoch(model, train_loader, optimizer)
            elapsed = perf_counter() - started
            correct, total = evaluate_counts(model, test_loader)
            row = ResultRow(
                method="single",
                workers=1,
                epoch=epoch,
                epoch_time_sec=elapsed,
                train_loss=loss_sum / sample_count,
                test_accuracy=100.0 * correct / total,
            )
            writer.write(row)
            print(
                f"[single] epoch={epoch} time={elapsed:.3f}s "
                f"loss={row.train_loss:.4f} accuracy={row.test_accuracy:.2f}%"
            )

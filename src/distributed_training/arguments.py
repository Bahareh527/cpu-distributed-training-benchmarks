"""Consistent command-line arguments for benchmark entry points."""

import argparse

from distributed_training.config import TrainingConfig


def add_training_arguments(parser: argparse.ArgumentParser, *, distributed: bool) -> None:
    if distributed:
        parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--dataset", choices=("synthetic", "mnist"), default="synthetic")
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--test-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--out", required=True)


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        epochs=args.epochs,
        global_batch_size=args.global_batch_size,
        learning_rate=args.learning_rate,
        dataset=args.dataset,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
        loader_workers=args.loader_workers,
        threads_per_worker=args.threads_per_worker,
    )

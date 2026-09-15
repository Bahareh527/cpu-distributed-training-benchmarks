"""Shared launcher for manual ring-allreduce and PyTorch DDP."""

import os
import socket
from datetime import timedelta
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from distributed_training.common import configure_runtime, evaluate_counts, train_one_epoch
from distributed_training.config import TrainingConfig
from distributed_training.data import build_datasets, make_loader, partition_indices
from distributed_training.model import DigitCNN
from distributed_training.results import ResultRow, ResultWriter
from distributed_training.ring import synchronize_model_gradients


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _summarize_epoch(
    loss_sum: float, sample_count: int, elapsed: float, correct: int, total: int
) -> tuple[float, float, float]:
    training = torch.tensor([loss_sum, float(sample_count)], dtype=torch.float64)
    timing = torch.tensor(elapsed, dtype=torch.float64)
    evaluation = torch.tensor([float(correct), float(total)], dtype=torch.float64)
    dist.all_reduce(training, op=dist.ReduceOp.SUM)
    dist.all_reduce(timing, op=dist.ReduceOp.MAX)
    dist.all_reduce(evaluation, op=dist.ReduceOp.SUM)
    return (
        float(timing.item()),
        float((training[0] / training[1]).item()),
        float((100.0 * evaluation[0] / evaluation[1]).item()),
    )


def _worker(
    rank: int,
    world_size: int,
    port: int,
    method: str,
    config: TrainingConfig,
    output: str,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    configure_runtime(config.seed, config.threads_per_worker)
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=90))
    writer = ResultWriter(output) if rank == 0 else None
    try:
        train_data, test_data = build_datasets(config)
        local_batch = config.local_batch_size(world_size)
        train_loader = make_loader(
            train_data,
            partition_indices(len(train_data), rank, world_size, equal=True),
            local_batch,
            shuffle=True,
            seed=config.seed + rank,
            loader_workers=config.loader_workers,
        )
        test_loader = make_loader(
            test_data,
            partition_indices(len(test_data), rank, world_size, equal=False),
            min(512, max(1, len(test_data) // world_size)),
            shuffle=False,
            seed=config.seed,
            loader_workers=config.loader_workers,
        )
        base_model = DigitCNN()
        if method == "ddp":
            training_model: torch.nn.Module = DistributedDataParallel(base_model)
        elif method == "ring":
            for parameter in base_model.parameters():
                dist.broadcast(parameter.data, src=0)
            training_model = base_model
        else:
            raise ValueError(f"unsupported synchronous method: {method}")
        optimizer = torch.optim.SGD(training_model.parameters(), lr=config.learning_rate)
        gradient_hook = synchronize_model_gradients if method == "ring" else None

        # Warm up the full process group before the ring's first batched P2P operation.
        dist.barrier()
        for epoch in range(1, config.epochs + 1):
            dist.barrier()
            started = perf_counter()
            loss_sum, sample_count = train_one_epoch(
                training_model,
                train_loader,
                optimizer,
                synchronize_gradients=gradient_hook,
            )
            elapsed = perf_counter() - started
            evaluation_model = base_model
            correct, total = evaluate_counts(evaluation_model, test_loader)
            epoch_time, train_loss, accuracy = _summarize_epoch(
                loss_sum, sample_count, elapsed, correct, total
            )
            if writer is not None:
                row = ResultRow(method, world_size, epoch, epoch_time, train_loss, accuracy)
                writer.write(row)
                print(
                    f"[{method}] workers={world_size} epoch={epoch} time={epoch_time:.3f}s "
                    f"loss={train_loss:.4f} accuracy={accuracy:.2f}%"
                )
    finally:
        if writer is not None:
            writer.close()
        dist.destroy_process_group()


def launch_synchronous(
    method: str, workers: int, config: TrainingConfig, output: str | Path
) -> None:
    config.validate(workers)
    if method not in {"ddp", "ring"}:
        raise ValueError("method must be 'ddp' or 'ring'")
    mp.spawn(
        _worker,
        args=(workers, _free_port(), method, config, str(output)),
        nprocs=workers,
        join=True,
    )

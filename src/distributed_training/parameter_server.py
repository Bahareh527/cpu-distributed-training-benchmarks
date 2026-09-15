"""Synchronous centralized parameter-server benchmark using local processes."""

import multiprocessing as mp
from collections import defaultdict
from multiprocessing.connection import Connection, wait
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from distributed_training.common import (
    configure_runtime,
    evaluate_counts,
    gradients_to_numpy,
    load_numpy_state,
    set_numpy_gradients,
    state_to_numpy,
)
from distributed_training.config import TrainingConfig
from distributed_training.data import build_datasets, make_loader, partition_indices
from distributed_training.model import DigitCNN
from distributed_training.results import ResultRow, ResultWriter


def average_gradients(gradient_sets: list[list[np.ndarray]]) -> list[np.ndarray]:
    if not gradient_sets:
        raise ValueError("at least one gradient set is required")
    expected = len(gradient_sets[0])
    if any(len(gradients) != expected for gradients in gradient_sets):
        raise ValueError("workers supplied different gradient counts")
    return [
        np.mean([gradients[index] for gradients in gradient_sets], axis=0)
        for index in range(expected)
    ]


def _server(connections: list[Connection], workers: int, config: TrainingConfig) -> None:
    configure_runtime(config.seed, config.threads_per_worker)
    model = DigitCNN()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    gradient_barriers: dict[tuple[int, int], list[tuple[Connection, list[np.ndarray]]]] = (
        defaultdict(list)
    )
    epoch_barriers: dict[int, list[tuple[Connection, float, float, int]]] = defaultdict(list)
    evaluation_barriers: dict[int, list[tuple[Connection, int, int]]] = defaultdict(list)
    finished = 0

    while finished < workers:
        for connection in wait(connections):
            message = connection.recv()
            kind = message[0]
            if kind == "pull":
                connection.send(("state", state_to_numpy(model)))
            elif kind == "gradients":
                _, epoch, step, gradients = message
                key = (epoch, step)
                gradient_barriers[key].append((connection, gradients))
                if len(gradient_barriers[key]) == workers:
                    synchronized = gradient_barriers.pop(key)
                    optimizer.zero_grad(set_to_none=True)
                    gradients = average_gradients([item[1] for item in synchronized])
                    set_numpy_gradients(model, gradients)
                    optimizer.step()
                    state = state_to_numpy(model)
                    for waiting_connection, _ in synchronized:
                        waiting_connection.send(("state", state))
            elif kind == "epoch":
                _, epoch, elapsed, loss_sum, sample_count = message
                epoch_barriers[epoch].append(
                    (connection, float(elapsed), float(loss_sum), int(sample_count))
                )
                if len(epoch_barriers[epoch]) == workers:
                    synchronized = epoch_barriers.pop(epoch)
                    maximum_time = max(item[1] for item in synchronized)
                    mean_loss = sum(item[2] for item in synchronized) / sum(
                        item[3] for item in synchronized
                    )
                    state = state_to_numpy(model)
                    for waiting_connection, *_ in synchronized:
                        waiting_connection.send(("epoch", state, maximum_time, mean_loss))
            elif kind == "evaluation":
                _, epoch, correct, total = message
                evaluation_barriers[epoch].append((connection, int(correct), int(total)))
                if len(evaluation_barriers[epoch]) == workers:
                    synchronized = evaluation_barriers.pop(epoch)
                    accuracy = (
                        100.0
                        * sum(item[1] for item in synchronized)
                        / sum(item[2] for item in synchronized)
                    )
                    for waiting_connection, *_ in synchronized:
                        waiting_connection.send(("accuracy", accuracy))
            elif kind == "done":
                finished += 1
                connections.remove(connection)
                connection.close()
            else:
                raise RuntimeError(f"unknown parameter-server message: {kind}")


def _worker(
    rank: int,
    workers: int,
    connection: Connection,
    config: TrainingConfig,
    output: str,
) -> None:
    configure_runtime(config.seed, config.threads_per_worker)
    train_data, test_data = build_datasets(config)
    train_loader = make_loader(
        train_data,
        partition_indices(len(train_data), rank, workers, equal=True),
        config.local_batch_size(workers),
        shuffle=True,
        seed=config.seed + rank,
        loader_workers=config.loader_workers,
    )
    test_loader = make_loader(
        test_data,
        partition_indices(len(test_data), rank, workers, equal=False),
        min(512, max(1, len(test_data) // workers)),
        shuffle=False,
        seed=config.seed,
        loader_workers=config.loader_workers,
    )
    model = DigitCNN()
    criterion = nn.CrossEntropyLoss()
    writer = ResultWriter(output) if rank == 0 else None
    try:
        connection.send(("pull",))
        _, initial_state = connection.recv()
        load_numpy_state(model, initial_state)

        for epoch in range(1, config.epochs + 1):
            model.train()
            loss_sum = 0.0
            sample_count = 0
            started = perf_counter()
            for step, (inputs, labels) in enumerate(train_loader):
                model.zero_grad(set_to_none=True)
                loss = criterion(model(inputs), labels)
                loss.backward()
                connection.send(("gradients", epoch, step, gradients_to_numpy(model)))
                _, state = connection.recv()
                load_numpy_state(model, state)
                batch_size = labels.numel()
                loss_sum += loss.item() * batch_size
                sample_count += batch_size
            elapsed = perf_counter() - started
            connection.send(("epoch", epoch, elapsed, loss_sum, sample_count))
            _, state, epoch_time, train_loss = connection.recv()
            load_numpy_state(model, state)

            correct, total = evaluate_counts(model, test_loader)
            connection.send(("evaluation", epoch, correct, total))
            _, accuracy = connection.recv()
            if writer is not None:
                row = ResultRow(
                    "parameter-server", workers, epoch, epoch_time, train_loss, accuracy
                )
                writer.write(row)
                print(
                    f"[parameter-server] workers={workers} epoch={epoch} time={epoch_time:.3f}s "
                    f"loss={train_loss:.4f} accuracy={accuracy:.2f}%"
                )
        connection.send(("done",))
    finally:
        if writer is not None:
            writer.close()
        connection.close()


def launch_parameter_server(workers: int, config: TrainingConfig, output: str | Path) -> None:
    """Launch one server process and ``workers`` training processes."""

    config.validate(workers)
    context = mp.get_context("spawn")
    pipe_pairs = [context.Pipe(duplex=True) for _ in range(workers)]
    server_connections = [pair[0] for pair in pipe_pairs]
    worker_connections = [pair[1] for pair in pipe_pairs]
    server = context.Process(target=_server, args=(server_connections, workers, config))
    processes = [
        context.Process(
            target=_worker,
            args=(rank, workers, worker_connections[rank], config, str(output)),
        )
        for rank in range(workers)
    ]
    server.start()
    for process in processes:
        process.start()
    for connection in server_connections + worker_connections:
        connection.close()
    for process in processes:
        process.join()
    server.join()
    exit_codes = [server.exitcode, *(process.exitcode for process in processes)]
    if any(code != 0 for code in exit_codes):
        raise RuntimeError(f"parameter-server process failure: exit codes {exit_codes}")

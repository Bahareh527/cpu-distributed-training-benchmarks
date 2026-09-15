"""Training, evaluation, and serialization shared by all methods."""

from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn


def configure_runtime(seed: int, threads: int) -> None:
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch allows this setting only before inter-op work starts.
        pass


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    synchronize_gradients: Callable[[nn.Module], None] | None = None,
) -> tuple[float, int]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    sample_count = 0
    for inputs, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), labels)
        loss.backward()
        if synchronize_gradients is not None:
            synchronize_gradients(model)
        optimizer.step()
        batch_size = labels.numel()
        loss_sum += loss.item() * batch_size
        sample_count += batch_size
    return loss_sum, sample_count


@torch.no_grad()
def evaluate_counts(
    model: nn.Module, loader: Iterable[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[int, int]:
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in loader:
        predictions = model(inputs).argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.numel()
    return correct, total


def state_to_numpy(model: nn.Module) -> dict[str, np.ndarray]:
    return {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}


def load_numpy_state(model: nn.Module, state: Mapping[str, np.ndarray]) -> None:
    tensors = {name: torch.from_numpy(value) for name, value in state.items()}
    model.load_state_dict(tensors)


def gradients_to_numpy(model: nn.Module) -> list[np.ndarray]:
    return [parameter.grad.detach().cpu().numpy().copy() for parameter in model.parameters()]


def set_numpy_gradients(model: nn.Module, gradients: Sequence[np.ndarray]) -> None:
    parameters = list(model.parameters())
    if len(parameters) != len(gradients):
        raise ValueError("gradient count does not match the model")
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = torch.from_numpy(gradient).to(dtype=parameter.dtype).clone()

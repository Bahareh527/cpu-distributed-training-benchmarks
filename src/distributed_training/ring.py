"""Manual ring-allreduce implemented with PyTorch point-to-point operations."""

import torch
import torch.distributed as dist
from torch import nn


def padded_chunk_size(element_count: int, world_size: int) -> int:
    if element_count < 0 or world_size < 1:
        raise ValueError("invalid chunk arguments")
    return (element_count + world_size - 1) // world_size


def flatten_gradients(model: nn.Module) -> torch.Tensor:
    gradients = [parameter.grad.reshape(-1) for parameter in model.parameters()]
    if not gradients:
        raise ValueError("model has no gradients")
    return torch.cat(gradients)


def assign_flat_gradients(model: nn.Module, flattened: torch.Tensor) -> None:
    offset = 0
    for parameter in model.parameters():
        count = parameter.numel()
        parameter.grad.copy_(flattened[offset : offset + count].view_as(parameter))
        offset += count
    if offset != flattened.numel():
        raise ValueError("flattened gradient has the wrong size")


def _exchange(
    send: torch.Tensor,
    receive: torch.Tensor,
    send_to: int,
    receive_from: int,
    tag: int,
) -> None:
    operations = [
        dist.P2POp(dist.isend, send.contiguous(), send_to, tag=tag),
        dist.P2POp(dist.irecv, receive, receive_from, tag=tag),
    ]
    for request in dist.batch_isend_irecv(operations):
        request.wait()


def ring_allreduce(tensor: torch.Tensor) -> torch.Tensor:
    """Return the average of ``tensor`` across ranks using a two-phase ring."""

    if not dist.is_initialized():
        raise RuntimeError("a process group must be initialized")
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor.clone()

    rank = dist.get_rank()
    original_size = tensor.numel()
    chunk_size = padded_chunk_size(original_size, world_size)
    padded_size = chunk_size * world_size
    flat = torch.zeros(padded_size, dtype=tensor.dtype, device=tensor.device)
    flat[:original_size].copy_(tensor.reshape(-1))
    flat.div_(world_size)
    chunks = list(flat.split(chunk_size))
    next_rank = (rank + 1) % world_size
    previous_rank = (rank - 1) % world_size

    # Reduce-scatter: rank r finishes with reduced chunk (r + 1) mod world_size.
    for step in range(world_size - 1):
        send_index = (rank - step) % world_size
        receive_index = (rank - step - 1) % world_size
        received = torch.empty_like(chunks[receive_index])
        _exchange(chunks[send_index], received, next_rank, previous_rank, tag=step)
        chunks[receive_index].add_(received)

    # All-gather: circulate the reduced chunks until every rank owns all of them.
    for step in range(world_size - 1):
        send_index = (rank - step + 1) % world_size
        receive_index = (rank - step) % world_size
        received = torch.empty_like(chunks[receive_index])
        _exchange(
            chunks[send_index],
            received,
            next_rank,
            previous_rank,
            tag=world_size + step,
        )
        chunks[receive_index].copy_(received)

    return flat[:original_size].clone()


def synchronize_model_gradients(model: nn.Module) -> None:
    assign_flat_gradients(model, ring_allreduce(flatten_gradients(model)))

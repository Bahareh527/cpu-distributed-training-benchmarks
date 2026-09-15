import torch

from distributed_training.model import DigitCNN


def test_model_output_shape() -> None:
    assert DigitCNN()(torch.zeros(4, 1, 28, 28)).shape == (4, 10)

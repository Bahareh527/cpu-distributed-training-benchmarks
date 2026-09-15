import numpy as np
import pytest

from distributed_training.parameter_server import average_gradients


def test_gradient_averaging() -> None:
    result = average_gradients([[np.array([1.0, 3.0])], [np.array([3.0, 5.0])]])
    assert result[0].tolist() == pytest.approx([2.0, 4.0])


def test_gradient_averaging_rejects_mismatched_models() -> None:
    with pytest.raises(ValueError, match="different gradient counts"):
        average_gradients([[np.ones(1)], [np.ones(1), np.ones(1)]])

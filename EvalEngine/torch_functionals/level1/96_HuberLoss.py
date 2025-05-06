import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the Smooth L1 (Huber) Loss for regression tasks.

    Args:
        predictions (torch.Tensor): Predicted values.
        targets (torch.Tensor): Target values.

    Returns:
        torch.Tensor: Smooth L1 (Huber) Loss.
    """
    return F.smooth_l1_loss(predictions, targets)


class Model(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks.

    Parameters:
        None
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets, fn=module_fn):
        return fn(predictions, targets)


batch_size = 128
input_shape = (4096,)
dim = 1


def get_inputs():
    return [
        torch.randn(batch_size, *input_shape),
        torch.randn(batch_size, *input_shape),
    ]


def get_init_inputs():
    return []

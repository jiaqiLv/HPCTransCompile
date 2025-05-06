import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor) -> torch.Tensor:
    """
    Applies Softmax activation to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features).

    Returns:
        torch.Tensor: Output tensor with Softmax applied, same shape as input.
    """
    return F.softmax(x, dim=1)


class Model(nn.Module):
    """
    Simple model that performs a Softmax activation.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(x)


batch_size = 16
dim = 16384


def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed

import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor) -> torch.Tensor:
    """
    Applies Sigmoid activation to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of any shape.

    Returns:
        torch.Tensor: Output tensor with Sigmoid applied, same shape as input.
    """
    return torch.sigmoid(x)


class Model(nn.Module):
    """
    Simple model that performs a Sigmoid activation.
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

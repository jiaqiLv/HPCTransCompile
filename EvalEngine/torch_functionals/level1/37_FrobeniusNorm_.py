import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor) -> torch.Tensor:
    """
    Applies Frobenius norm normalization to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of arbitrary shape.

    Returns:
        torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
    """
    norm = torch.norm(x, p="fro")
    return x / norm


class Model(nn.Module):
    """
    Simple model that performs Frobenius norm normalization.
    """

    def __init__(self):
        """
        Initializes the Frobenius norm normalization layer.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Applies Frobenius norm normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.
            fn (callable): Function to apply normalization, defaults to module_fn

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        return fn(x)


batch_size = 16
features = 64
dim1 = 256
dim2 = 256


def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    return []

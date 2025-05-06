import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Applies LogSoftmax activation to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, dim)
        dim (int): Dimension along which to apply LogSoftmax

    Returns:
        torch.Tensor: Output tensor with LogSoftmax applied, same shape as input
    """
    return F.log_softmax(x, dim=dim)


class Model(nn.Module):
    """
    Simple model that performs a LogSoftmax activation.
    """

    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(x, self.dim)


batch_size = 16
dim = 16384
sm_dim = 1


def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]


def get_init_inputs():
    return [sm_dim]

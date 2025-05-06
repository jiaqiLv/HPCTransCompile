import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs a single matrix multiplication (C = A * B) with irregular shapes.

    Args:
        A: Input tensor with shape (M, K).
        B: Input tensor with shape (K, N).

    Returns:
        C: Output tensor with shape (M, N).
    """
    return torch.matmul(A, B)


class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with irregular shapes
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(A, B)


M = 8205
K = 2949
N = 5921


def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed

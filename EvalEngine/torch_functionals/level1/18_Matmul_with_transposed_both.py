import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Performs a single matrix multiplication with transposed A and B (C = A.T * B.T).

    Args:
        A: Input tensor of shape (K, M).
        B: Input tensor of shape (N, K).

    Returns:
        Output tensor of shape (M, N).
    """
    return torch.matmul(A.T, B.T)


class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor, fn=module_fn) -> torch.Tensor:
        return fn(A, B)


M = 1024
K = 4096
N = 2048


def get_inputs():
    A = torch.randn(K, M)
    B = torch.randn(N, K)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed

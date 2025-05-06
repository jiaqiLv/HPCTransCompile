import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
) -> torch.Tensor:
    """
    Functional Group Normalization.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).
        weight (torch.Tensor): Weight tensor of shape (num_features).
        bias (torch.Tensor): Bias tensor of shape (num_features).
        num_groups (int): Number of groups to divide the channels into.
        eps (float): Epsilon parameter for numerical stability.

    Returns:
        torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
    """
    return F.group_norm(x, num_groups=num_groups, weight=weight, bias=bias, eps=eps)


class Model(nn.Module):
    """
    Simple model that performs Group Normalization.
    """

    def __init__(self, num_features: int, num_groups: int, eps: float):
        """
        Initializes the GroupNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            num_groups (int): Number of groups to divide the channels into.
        """
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        return fn(x, self.weight, self.bias, num_groups, self.eps)


batch_size = 16
features = 64
num_groups = 8
dim1 = 256
dim2 = 256
eps = 1e-5


def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]


def get_init_inputs():
    return [features, num_groups, eps]

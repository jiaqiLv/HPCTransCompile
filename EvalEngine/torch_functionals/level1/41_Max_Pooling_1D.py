import torch
import torch.nn as nn
import torch.nn.functional as F


def module_fn(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    return_indices: bool,
) -> torch.Tensor:
    """
    Functional implementation of Max Pooling 1D.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).
        kernel_size (int): Size of the window to take a max over.
        stride (int): Stride of the window.
        padding (int): Implicit zero padding to be added on both sides.
        dilation (int): Spacing between kernel elements.
        return_indices (bool): Whether to return the indices of the maximum values.

    Returns:
        torch.Tensor: Output tensor with Max Pooling 1D applied.
    """
    return F.max_pool1d(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_indices=return_indices,
    )


class Model(nn.Module):
    """
    Simple model that performs Max Pooling 1D.
    """

    def __init__(
        self,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        return_indices: bool,
    ):
        """
        Initializes the Max Pooling 1D layer.

        Args:
            kernel_size (int): Size of the window to take a max over.
            stride (int): Stride of the window.
            padding (int): Implicit zero padding to be added on both sides.
            dilation (int): Spacing between kernel elements.
            return_indices (bool): Whether to return the indices of the maximum values.
        """
        super(Model, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor, fn=module_fn) -> torch.Tensor:
        """
        Applies Max Pooling 1D to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).
            fn: Function to apply (defaults to module_fn)

        Returns:
            torch.Tensor: Output tensor with Max Pooling 1D applied.
        """
        return fn(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.return_indices,
        )


batch_size = 16
features = 64
sequence_length = 128
kernel_size = 4
stride = 2
padding = 2
dilation = 3
return_indices = False


def get_inputs():
    x = torch.randn(batch_size, features, sequence_length)
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]

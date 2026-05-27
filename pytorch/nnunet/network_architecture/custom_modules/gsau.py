from torch import nn
import torch


DEFAULT_GSAU_KERNEL_SIZE = 7
DEFAULT_GSAU_INIT_BIAS = 2.0
DEFAULT_GSAU_WEIGHT_STD = 1e-3


class GSAU(nn.Module):
    """Sigmoid-gated semantic skip gate inspired by MAN GSAU/SGAB."""

    def __init__(
        self,
        channels,
        kernel_size=DEFAULT_GSAU_KERNEL_SIZE,
        init_bias=DEFAULT_GSAU_INIT_BIAS,
        weight_std=DEFAULT_GSAU_WEIGHT_STD,
    ):
        super().__init__()
        if not isinstance(channels, int) or channels <= 0:
            raise ValueError("GSAU channels must be a positive integer.")
        if not isinstance(kernel_size, int) or kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("GSAU kernel_size must be a positive odd integer.")
        if weight_std < 0:
            raise ValueError("GSAU weight_std must be non-negative.")

        self.channels = channels
        self.kernel_size = kernel_size
        self.init_bias = init_bias
        self.weight_std = weight_std
        self.output_channels = channels

        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=channels,
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.dwconv.weight, mean=0.0, std=self.weight_std)
        nn.init.constant_(self.dwconv.bias, self.init_bias)

    def compute_gate(self, gate_x):
        if gate_x.dim() != 4:
            raise ValueError("GSAU expects gate_x shaped [B, C, H, W].")
        if gate_x.size(1) != self.channels:
            raise ValueError("Expected %d gate channels, got %d." % (self.channels, gate_x.size(1)))
        return torch.sigmoid(self.dwconv(gate_x))

    def forward(self, gate_x, target_y):
        if gate_x.shape != target_y.shape:
            raise ValueError("GSAU requires gate_x and target_y to have the same shape.")
        return target_y * self.compute_gate(gate_x)

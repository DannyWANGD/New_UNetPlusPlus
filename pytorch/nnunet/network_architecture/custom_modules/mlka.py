import torch
from torch import nn
import torch.nn.functional as F


MLKA_NUM_BRANCHES = 3
MLKA_KERNEL_CONFIGS = (
    (3, 5, 2),
    (5, 7, 3),
    (7, 9, 4),
)


class LayerNorm(nn.Module):
    """LayerNorm supporting channels_last and channels_first tensors."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ("channels_last", "channels_first"):
            raise NotImplementedError("LayerNorm only supports channels_last and channels_first.")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def _make_norm(channels, norm_type):
    if norm_type == "instance":
        return nn.InstanceNorm2d(channels, eps=1e-5, affine=True)
    if norm_type == "batch":
        return nn.BatchNorm2d(channels, eps=1e-5, affine=True)
    if norm_type == "layer":
        return LayerNorm(channels, data_format="channels_first")
    raise ValueError("Unknown MLKA norm_type: %s" % norm_type)


class _LKABranch(nn.Sequential):
    """One MAN-style large-kernel attention branch."""

    def __init__(self, channels, base_kernel, dilated_kernel, dilation):
        super().__init__(
            nn.Conv2d(
                channels,
                channels,
                base_kernel,
                stride=1,
                padding=base_kernel // 2,
                groups=channels,
            ),
            nn.Conv2d(
                channels,
                channels,
                dilated_kernel,
                stride=1,
                padding=(dilated_kernel // 2) * dilation,
                dilation=dilation,
                groups=channels,
            ),
            nn.Conv2d(channels, channels, 1, stride=1, padding=0),
        )


class GroupGLKA(nn.Module):
    """Configurable 2D GroupGLKA adapted from MAN for UNet++ refinement nodes."""

    def __init__(self, channels, num_groups=MLKA_NUM_BRANCHES, norm_type="instance"):
        super().__init__()
        if num_groups != MLKA_NUM_BRANCHES:
            raise NotImplementedError("Current GroupGLKA implementation supports exactly 3 groups.")
        if channels % num_groups != 0:
            raise ValueError("channels must be divisible by num_groups for GroupGLKA.")

        branch_channels = channels // num_groups
        if branch_channels < 8:
            raise ValueError("Each GroupGLKA branch requires at least 8 channels.")

        self.channels = channels
        self.num_groups = num_groups
        self.output_channels = channels

        self.norm = _make_norm(channels, norm_type)
        self.scale = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

        self.lka_branches = nn.ModuleList(
            [
                _LKABranch(branch_channels, base_kernel, dilated_kernel, dilation)
                for base_kernel, dilated_kernel, dilation in MLKA_KERNEL_CONFIGS
            ]
        )
        self.dwconv_branches = nn.ModuleList(
            [
                nn.Conv2d(
                    branch_channels,
                    branch_channels,
                    base_kernel,
                    stride=1,
                    padding=base_kernel // 2,
                    groups=branch_channels,
                )
                for base_kernel, _, _ in MLKA_KERNEL_CONFIGS
            ]
        )

        self.proj_first = nn.Conv2d(channels, channels * 2, 1, stride=1, padding=0)
        self.proj_last = nn.Conv2d(channels, channels, 1, stride=1, padding=0)

    def forward(self, x, pre_attn=None, RAA=None):
        if x.dim() != 4:
            raise ValueError("GroupGLKA expects a 4D tensor shaped [B, C, H, W].")
        if x.size(1) != self.channels:
            raise ValueError("Expected %d channels, got %d." % (self.channels, x.size(1)))

        shortcut = x
        x = self.proj_first(self.norm(x))
        attention, features = torch.chunk(x, 2, dim=1)
        attention_chunks = torch.chunk(attention, self.num_groups, dim=1)

        attention = torch.cat(
            [
                lka_branch(attention_chunk) * dwconv_branch(attention_chunk)
                for lka_branch, dwconv_branch, attention_chunk in zip(
                    self.lka_branches, self.dwconv_branches, attention_chunks
                )
            ],
            dim=1,
        )

        return self.proj_last(features * attention) * self.scale + shortcut


class MLKABlock(GroupGLKA):
    """UNet++ node refinement block based on MAN GroupGLKA."""

    def __init__(self, channels, num_groups=MLKA_NUM_BRANCHES, norm_type="instance"):
        super().__init__(channels=channels, num_groups=num_groups, norm_type=norm_type)

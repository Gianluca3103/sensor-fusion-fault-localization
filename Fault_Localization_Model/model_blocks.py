import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


def resize_reliability_map(tensor, size):
    """Resize continuous reliability values without nearest-sample aliasing."""
    if size is None or tensor.shape[-2:] == tuple(size):
        return tensor
    old_height, old_width = tensor.shape[-2:]
    new_height, new_width = tuple(size)
    if new_height <= old_height and new_width <= old_width:
        return F.interpolate(tensor, size=(new_height, new_width), mode="area")
    return F.interpolate(tensor, size=(new_height, new_width), mode="nearest")


def frozen_reference_forward(module, *args, **kwargs):
    """Run a no-gradient target branch without updating normalization state."""
    training_states = [
        (submodule, submodule.training) for submodule in module.modules()
    ]
    try:
        module.eval()
        with torch.no_grad():
            return module(*args, **kwargs)
    finally:
        for submodule, was_training in training_states:
            submodule.training = was_training

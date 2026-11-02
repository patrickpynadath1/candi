"""Per-position squared error in a continuous target representation."""

from torch import nn


class LatentMSE(nn.Module):
    def forward(self, prediction, target):
        return (prediction.float() - target.float()).square().mean(-1)

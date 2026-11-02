"""Replaceable second sublayers for DiT blocks, independent of corruption."""

import torch
from torch import nn

from .layers import regular_attention_multi_headed, rotate_half


class PositionwiseMLP(nn.Sequential):
    """Keep the historical numbered parameter names for checkpoint compatibility."""

    def __init__(self, dim, expansion=4, hidden_size=None):
        width = expansion * dim if hidden_size is None else hidden_size
        if not isinstance(width, int) or width < 1:
            raise ValueError("MLP hidden size must be a positive integer")
        super().__init__(
            nn.Linear(dim, width),
            nn.GELU(approximate="tanh"),
            nn.Linear(width, dim),
        )

    def forward(self, x, *, rotary_cos_sin=None, attention_mask=None):
        return super().forward(x)


class ExpandedAttention(nn.Module):
    """Expand Q/K/V width, mix positions, then project to the residual width.

    Retain the first attention's head dimension so the block's positional
    encoding is shared exactly. Expansion increases the number of heads.
    Normalization, time modulation, dropout and residual gating belong to the
    enclosing block, just as they do for its MLP.
    """

    def __init__(self, dim, n_heads, expansion=2):
        super().__init__()
        if not isinstance(expansion, int) or expansion < 1:
            raise ValueError("attention expansion must be a positive integer")
        if dim % n_heads or (dim // n_heads) % 2:
            raise ValueError("attention requires an even integral head dimension")
        self.n_heads = n_heads * expansion
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim * expansion, bias=False)
        self.output = nn.Linear(dim * expansion, dim, bias=False)

    def forward(self, x, *, rotary_cos_sin, attention_mask=None):
        q, k, v = (
            self.qkv(x)
            .reshape(*x.shape[:-1], 3, self.n_heads, self.head_dim)
            .unbind(-3)
        )
        cos, sin = (r[:, :, 0].to(q.dtype) for r in rotary_cos_sin)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return self.output(
            regular_attention_multi_headed(q, k, v, attention_mask=attention_mask)
        )


class CenteredMLP(nn.Module):
    """Subtract a gated sequence mean before GELU, using one scalar per block.

    Full, unpadded sequences only: a global mean would leak across causal or
    padding masks. Cached execution is rejected by the enclosing DiT block.
    """

    def __init__(self, dim, n_heads=None, expansion=4):
        super().__init__()
        self.up = nn.Linear(dim, expansion * dim)
        self.activation = nn.GELU(approximate="tanh")
        self.down = nn.Linear(expansion * dim, dim)
        self.centering_strength = nn.Parameter(torch.zeros(()))

    def forward(self, x, *, rotary_cos_sin=None, attention_mask=None):
        if attention_mask is not None:
            raise ValueError("CenteredMLP requires full unmasked sequences")
        a = self.up(x)
        mean = a.mean(dim=-2, keepdim=True)
        centered = a - self.centering_strength.to(a.dtype) * mean
        return self.down(self.activation(centered))

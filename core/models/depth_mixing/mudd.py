"""MUDDFormer depth aggregation (arXiv:2502.12170, Eq. 6 / Appendix A)."""

import torch
from torch import nn


class MUDDDepthMixer(nn.Module):
    """Route complete block outputs to Q, K, V and attention residual streams.

    Each position's latest state generates unconstrained, independent weights
    over earlier depths for all four streams. No tokens are mixed here. The
    shared MLP follows the paper's joint implementation, with width 4*S.
    Initialization selects the latest state exactly, recovering the baseline.

    Inputs are [batch, positions, sources, dim]; outputs are
    [batch, positions, 4, dim], ordered Q, K, V, residual. The enclosing model
    owns history and attention normalization. The paper also applies a final
    mixer after the last block and returns its residual stream.
    """

    def __init__(self, dim: int, num_sources: int):
        super().__init__()
        if dim < 1 or num_sources < 1:
            raise ValueError("dim and num_sources must be positive")
        self.dim = dim
        self.num_sources = num_sources
        width = 4 * num_sources
        self.norm = nn.RMSNorm(dim, eps=1e-6)
        self.up = nn.Linear(dim, width, bias=False)
        self.activation = nn.GELU()
        self.down = nn.Linear(width, width, bias=False)
        self.static_weights = nn.Parameter(torch.zeros(4, num_sources))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.norm.weight)
        nn.init.normal_(self.up.weight, std=self.dim**-0.5)
        nn.init.zeros_(self.down.weight)
        with torch.no_grad():
            self.static_weights.zero_()
            self.static_weights[:, -1] = 1

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 4 or history.shape[-2:] != (self.num_sources, self.dim):
            raise ValueError(
                "history must have shape [batch, positions, num_sources, dim]"
            )
        latest = history[:, :, -1]
        dynamic = self.down(self.activation(self.up(self.norm(latest))))
        weights = dynamic.reshape(*history.shape[:2], 4, self.num_sources)
        weights = weights + self.static_weights.to(weights.dtype)
        return torch.einsum("bncs,bnsd->bncd", weights.to(history.dtype), history)

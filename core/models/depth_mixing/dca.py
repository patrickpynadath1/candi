"""DeepCrossAttention's source-conditioned GRN-v3 aggregation.

Reference: https://arxiv.org/html/2502.06785v2, Sections 3.1–3.2.
"""

import math

import torch
from torch import nn


class DCA(nn.Module):
    """Mix residual contributions independently for Q, K and V at each token.

    ``history`` is [batch, position, source, feature], containing the original
    input followed by *incremental block contributions*, not full block states.
    The returned [batch, position, 4, feature] streams are Q, K, V, Q: Figure 4
    uses the query mixture as the attention residual entering the feedforward
    sublayer. Append attention + feedforward contributions to history afterward,
    without adding this query mixture a second time.

    The paper initializes static weights to one and dynamic gate vectors to
    zero. Standard PyTorch ReLU has zero derivative at zero, so that exact
    initialization leaves the dynamic gates inactive. ``gate_init_std`` makes a
    deliberate nonzero initialization available without altering the equation.
    A value such as 1e-3 lets the dynamic gates learn from the first update.
    """

    def __init__(self, dim: int, num_sources: int, gate_init_std: float = 0.0):
        super().__init__()
        if dim < 1 or num_sources < 1:
            raise ValueError("dim and num_sources must be positive")
        if not math.isfinite(gate_init_std) or gate_init_std < 0:
            raise ValueError("gate_init_std must be finite and nonnegative")
        self.dim = dim
        self.num_sources = num_sources
        self.static_weights = nn.Parameter(torch.ones(3, num_sources, dim))
        self.gate_vectors = nn.Parameter(torch.zeros(3, dim))
        if gate_init_std:
            nn.init.normal_(self.gate_vectors, std=gate_init_std)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 4 or history.shape[-2:] != (self.num_sources, self.dim):
            raise ValueError(
                f"DCA expects history [batch, position, {self.num_sources}, {self.dim}]"
            )
        dynamic = torch.einsum(
            "bnsd,rd->bnrs", history, self.gate_vectors.to(history.dtype)
        ).relu()
        weights = self.static_weights.to(history.dtype) + dynamic.unsqueeze(-1)
        mixed = (history.unsqueeze(-3) * weights).sum(dim=-2)
        return torch.cat((mixed, mixed[:, :, :1]), dim=-2)

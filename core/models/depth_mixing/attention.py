"""Content-based retrieval across depth, independently at each position."""

import torch
from torch import Tensor, nn


class DepthAttention(nn.Module):
    """Produce Q/K/V/residual inputs from a stack of earlier hidden states.

    ``history`` has shape [batch, position, source, dim], oldest first. Each
    stream uses the latest state as its query and all source states as keys.
    Values are the original hidden states, so retrieval adds no full-width
    value projections. Attention never crosses the position axis.

    A zero-initialized, unconstrained gate per stream preserves the latest
    state exactly at initialization. The gates learn first; query/key weights
    receive gradients once their gate becomes nonzero.
    """

    def __init__(
        self,
        dim: int,
        num_sources: int,
        key_dim: int = 32,
        gate_init: float = 0.0,
    ):
        super().__init__()
        if dim < 1 or num_sources < 1 or key_dim < 1:
            raise ValueError("dim, num_sources, and key_dim must be positive")
        self.dim = dim
        self.num_sources = num_sources
        self.key_dim = key_dim
        self.norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=False)
        self.query = nn.Linear(dim, 4 * key_dim, bias=False)
        self.key = nn.Linear(dim, 4 * key_dim, bias=False)
        self.gate = nn.Parameter(torch.full((4,), float(gate_init)))

    def forward(self, history: Tensor) -> Tensor:
        if history.ndim != 4 or history.shape[-2:] != (self.num_sources, self.dim):
            raise ValueError(
                "history must have shape [batch, position, "
                f"{self.num_sources}, {self.dim}]"
            )
        normalized = self.norm(history)
        query = self.query(normalized[..., -1, :]).unflatten(-1, (4, self.key_dim))
        key = self.key(normalized).unflatten(-1, (4, self.key_dim))
        scores = torch.einsum("bnrk,bnsrk->bnrs", query, key) * self.key_dim**-0.5
        weights = scores.float().softmax(dim=-1).to(history.dtype)
        retrieved = torch.einsum("bnrs,bnsd->bnrd", weights, history)
        latest = history[..., -1, :].unsqueeze(-2)
        return latest + self.gate.to(history.dtype)[None, None, :, None] * (
            retrieved - latest
        )

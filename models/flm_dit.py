"""DiT backbone for raw continuous vocabulary-space FLM states."""

import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    DDiTBlock,
    DDiTFinalLayer,
    LayerNorm,
    Rotary,
    TimestepEmbedder,
    modulate,
    split_and_apply_rotary_pos_emb,
)
from .registry import models


class FLMEmbedding(nn.Module):
    def __init__(self, dim, vocab_size):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(vocab_size, dim))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        if x.ndim == 2:
            return self.embedding[x]
        if x.ndim != 3:
            raise ValueError("FLM input must have shape (B,L) or (B,L,V)")
        return torch.einsum("blv,vd->bld", x.float(), self.embedding.float()).to(
            x.dtype
        )


class LearnableLossWeighting(nn.Module):
    def __init__(self, cond_dim, hidden_dim=128):
        super().__init__()
        self.time = TimestepEmbedder(cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, t):
        return self.mlp(self.time(t)).squeeze(-1)


class SoftcapDDiTBlock(nn.Module):
    """DDiT block with tanh-capped attention scores."""

    def __init__(self, dim, heads, cond_dim, dropout, softcap):
        super().__init__()
        self.heads, self.softcap, self.dropout = heads, softcap, dropout
        self.norm1, self.norm2 = LayerNorm(dim), LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * dim, dim),
        )
        self.modulation = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x, rotary, c):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(c).chunk(
            6, -1
        )
        hidden = modulate(self.norm1(x), shift_a[:, None], scale_a[:, None])
        qkv = einops.rearrange(
            self.qkv(hidden), "b l (q h d) -> b l q h d", q=3, h=self.heads
        )
        q, k, v = split_and_apply_rotary_pos_emb(qkv, rotary)
        scores = torch.einsum("blhd,bshd->bhls", q, k) / math.sqrt(q.shape[-1])
        scores = self.softcap * torch.tanh(scores / self.softcap)
        attention = torch.einsum("bhls,bshd->blhd", scores.softmax(-1), v).flatten(-2)
        x = x + gate_a[:, None] * F.dropout(
            self.out(attention), self.dropout, self.training
        )
        hidden = modulate(self.norm2(x), shift_m[:, None], scale_m[:, None])
        return x + gate_m[:, None] * F.dropout(
            self.mlp(hidden), self.dropout, self.training
        )


@models.register("flm_dit")
class FLMDIT(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        dim, cond_dim = config.model.hidden_size, config.model.cond_dim
        self.vocab_embed = FLMEmbedding(dim, vocab_size)
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.sigma_map_prime = (
            TimestepEmbedder(cond_dim)
            if getattr(config.algo, "double_temb", False)
            else None
        )
        self.learnable_loss_weighting = (
            LearnableLossWeighting(cond_dim)
            if getattr(config.algo, "learnable_loss_weighting", False)
            else None
        )
        self.rotary_emb = Rotary(dim // config.model.n_heads)
        softcap = getattr(config.model, "softcap", 0.0)
        block_type = SoftcapDDiTBlock if softcap > 0 else DDiTBlock
        self.blocks = nn.ModuleList()
        for _ in range(config.model.n_blocks):
            if softcap > 0:
                block = block_type(
                    dim, config.model.n_heads, cond_dim, config.model.dropout, softcap
                )
            else:
                block = block_type(
                    dim,
                    config.model.n_heads,
                    adaLN=True,
                    cond_dim=cond_dim,
                    dropout=config.model.dropout,
                )
            self.blocks.append(block)
        self.output_layer = DDiTFinalLayer(dim, vocab_size, cond_dim, True)

    def forward(self, x, sigma, sigma_prime=None):
        x = self.vocab_embed(x)
        cond = self.sigma_map(sigma)
        if sigma_prime is not None:
            cond = cond + (self.sigma_map_prime or self.sigma_map)(sigma_prime)
        cond = F.silu(cond)
        rotary = self.rotary_emb(x)
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
        ):
            for block in self.blocks:
                x = block(x, rotary, c=cond)
            return self.output_layer(x, c=cond)

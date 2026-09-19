"""Non-causal DiT for grid-native Flow Reasoning Models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DDiTBlock, DDiTFinalLayer, Rotary, TimestepEmbedder
from .flm_dit import FLMEmbedding
from .registry import models


@models.register("frm_dit")
class FRMDIT(nn.Module):
    """Embed the flow state, fixed puzzle, and recurrent carry separately."""

    def __init__(self, config, vocab_size):
        super().__init__()
        dim = config.model.hidden_size
        cond_dim = config.model.cond_dim
        self.vocab_size = vocab_size
        self.state_projection = FLMEmbedding(dim, vocab_size)
        self.carry_projection = FLMEmbedding(dim, vocab_size)
        self.clue_embedding = nn.Embedding(vocab_size, dim)
        self.clue_mask_embedding = nn.Embedding(2, dim)
        self.time_embedding = TimestepEmbedder(cond_dim)
        self.rotary_embedding = Rotary(dim // config.model.n_heads)
        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    dim,
                    config.model.n_heads,
                    adaLN=True,
                    cond_dim=cond_dim,
                    dropout=config.model.dropout,
                )
                for _ in range(config.model.n_blocks)
            ]
        )
        self.output_layer = DDiTFinalLayer(dim, vocab_size, cond_dim, True)

    def forward(self, state, time, clues, clue_mask, carry=None):
        if state.ndim != 3 or state.shape[-1] != self.vocab_size:
            raise ValueError("FRM state must have shape [batch, 81, vocabulary]")
        if clues.shape != state.shape[:2] or clue_mask.shape != clues.shape:
            raise ValueError("clues and clue_mask must match the FRM grid")
        if carry is None:
            carry = torch.zeros_like(state)
        if carry.shape != state.shape:
            raise ValueError("FRM carry must match the continuous state")

        hidden = (
            self.state_projection(state)
            + self.carry_projection(carry)
            + self.clue_embedding(clues)
            + self.clue_mask_embedding(clue_mask.long())
        )
        condition = F.silu(self.time_embedding(time))
        rotary = self.rotary_embedding(hidden)
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=hidden.is_cuda
        ):
            for block in self.blocks:
                hidden = block(hidden, rotary, c=condition)
            return self.output_layer(hidden, c=condition)

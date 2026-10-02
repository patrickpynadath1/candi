"""Noncausal DiT with legacy-compatible parameter names and initialization."""

import torch
from torch import nn
from torch.nn import functional as F

from .prediction import ModelPrediction, make_conditioning

from .layers import DDiTBlock, DDiTFinalLayer, EmbeddingLayer, Rotary, TimestepEmbedder


class DiT(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size=512,
        cond_dim=128,
        n_blocks=8,
        n_heads=8,
        dropout=0.1,
        adaLN=False,
        tie_word_embeddings=False,
        self_conditioning=False,
        self_conditioning_source="posterior_mean",
    ):
        super().__init__()
        if hidden_size % n_heads or (hidden_size // n_heads) % 2:
            raise ValueError("hidden_size must give an even integral head dimension")
        self.adaLN = adaLN
        self.vocab_embed = EmbeddingLayer(hidden_size, vocab_size)
        # Retain the unused time embedder for exact baseline checkpoint loading.
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(hidden_size // n_heads)
        self.blocks = nn.ModuleList(
            [
                DDiTBlock(hidden_size, n_heads, adaLN, cond_dim, dropout=dropout)
                for _ in range(n_blocks)
            ]
        )
        self.output_layer = DDiTFinalLayer(hidden_size, vocab_size, cond_dim, adaLN)
        if tie_word_embeddings:
            self.output_layer.linear.weight = self.vocab_embed.embedding

        self.conditioning_adapter = (
            make_conditioning(self_conditioning_source, hidden_size, hidden_size)
            if self_conditioning
            else None
        )

    def forward(
        self, tokens, conditioning, *, self_condition=None, return_features=False,
        attention_mask=None, position_ids=None
    ):
        with torch.autocast(tokens.device.type, enabled=False):
            embedded = self.vocab_embed(tokens)
        return self.forward_with_embeddings(
            embedded,
            conditioning,
            self_condition=self_condition,
            return_features=return_features,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    def forward_with_embeddings(
        self, hidden, conditioning, *, self_condition=None, return_features=False,
        attention_mask=None, position_ids=None
    ):
        hidden, condition = self.prepare_hidden(
            hidden, conditioning, self_condition=self_condition
        )
        return self.forward_hidden(
            hidden, condition, return_features=return_features,
            attention_mask=attention_mask, position_ids=position_ids
        )

    def prepare_hidden(self, hidden, conditioning, *, self_condition=None):
        # The original forward prepares embeddings and rotary frequencies in
        # FP32 before entering attention's BF16 context. In particular an
        # autocast einsum here would permanently cache rounded RoPE angles.
        with torch.autocast(hidden.device.type, enabled=False):
            if self.conditioning_adapter is not None:
                hidden = self.conditioning_adapter(hidden, self_condition)
            elif self_condition is not None:
                raise ValueError("model does not enable self-conditioning")
            condition = F.silu(self.sigma_map(conditioning)) if self.adaLN else None
        return hidden, condition

    def forward_hidden(self, hidden, condition, *, return_features=False,
                       attention_mask=None, position_ids=None):
        with torch.autocast(hidden.device.type, enabled=False):
            rotary = self.rotary_emb(hidden, position_ids=position_ids)
        # Precision is selected by the trainer or evaluation entrypoint.
        for block in self.blocks:
            hidden = block(hidden, rotary, condition, attention_mask)
        logits = self.project_hidden(hidden, condition)
        return ModelPrediction(logits, hidden) if return_features else logits

    def project_hidden(self, hidden, condition):
        return self.output_layer(hidden, condition)

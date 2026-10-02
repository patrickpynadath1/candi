"""Tokenwise embedding codebook with explicit geometry and decoding."""

import torch
from torch import nn
from torch.nn import functional as F

from .spec import RepresentationSpec


class EmbeddingRepresentation(nn.Module):
    def __init__(
        self,
        vocab_size,
        dimension=128,
        unit_normalized=True,
        trainable=False,
        project_after_update=False,
    ):
        super().__init__()
        if vocab_size < 2 or dimension < 2:
            raise ValueError("vocabulary and embedding dimension must be at least two")
        if project_after_update and (not trainable or not unit_normalized):
            raise ValueError(
                "post-update projection requires trainable unit embeddings"
            )
        self.project_after_update = project_after_update
        self.spec = RepresentationSpec(
            "embedding", dimension, vocab_size, unit_normalized
        )
        table = torch.randn(vocab_size, dimension) / dimension**0.5
        if unit_normalized:
            table = F.normalize(table, dim=-1)
        if trainable:
            self.table = nn.Parameter(table)
        else:
            self.register_buffer("table", table)

    def codebook(self):
        # Enforce the declared geometry on every use, including after optimization.
        return (
            F.normalize(self.table, dim=-1) if self.spec.unit_normalized else self.table
        )

    def encode(self, tokens):
        return F.embedding(tokens, self.codebook())

    def posterior_mean(self, probabilities):
        with torch.autocast(probabilities.device.type, enabled=False):
            return probabilities.float() @ self.codebook().float()

    def decode(self, state):
        # Dot-product decoding. For unit rows this is also nearest-neighbor decoding.
        with torch.autocast(state.device.type, enabled=False):
            scores = state.float() @ self.codebook().float().T
        return scores.argmax(-1)

    @torch.no_grad()
    def post_optimizer_step(self):
        if self.project_after_update:
            self.table.copy_(F.normalize(self.table, dim=-1))

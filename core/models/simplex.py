"""Simplex input mapping sharing the ordinary DiT transformer and output head."""

import torch
from .dit import DiT


class SimplexDiT(DiT):
    def __init__(self, vocab_size, input_mode="expectation", **kwargs):
        if input_mode not in {"expectation", "argmax"}:
            raise ValueError("simplex input_mode must be expectation or argmax")
        super().__init__(vocab_size, **kwargs)
        self.input_mode = input_mode
        self.vocab_size = vocab_size

    def embed_state(self, log_state):
        with torch.autocast(log_state.device.type, enabled=False):
            if self.input_mode == "argmax":
                embedded = self.vocab_embed(log_state.argmax(-1))
                return embedded.masked_fill(
                    ~torch.isfinite(log_state).any(-1)[..., None], 0
                )
            probabilities = log_state.exp().to(self.vocab_embed.embedding.dtype)
            return probabilities @ self.vocab_embed.embedding

    def forward(self, log_state, time, *, self_condition=None, return_features=False,
                attention_mask=None, position_ids=None):
        return self.forward_with_embeddings(
            self.embed_state(log_state),
            time.float(),
            self_condition=self_condition,
            return_features=return_features,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

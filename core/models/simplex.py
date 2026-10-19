"""Simplex input mapping sharing the ordinary DiT transformer and output head."""

import torch
from .dit import DiT
from .components import SimplexExpectationInput, SimplexArgmaxInput


class SimplexDiT(DiT):
    def __init__(
        self,
        vocab_size,
        input_mode="expectation",
        projection_precision="legacy",
        projection_chunk_size=0,
        **kwargs,
    ):
        if input_mode not in {"expectation", "argmax"}:
            raise ValueError("simplex input_mode must be expectation or argmax")
        super().__init__(vocab_size, **kwargs)
        if kwargs.get("input_adapter") is None:
            self.input_adapter = (
                SimplexExpectationInput(projection_precision, projection_chunk_size)
                if input_mode == "expectation"
                else SimplexArgmaxInput()
            )
        self.input_mode = input_mode
        self.vocab_size = vocab_size

    def embed_state(self, log_state):
        with torch.autocast(log_state.device.type, enabled=False):
            return self.input_adapter(log_state, self.vocab_embed)

    def forward(
        self,
        log_state,
        time,
        *,
        self_condition=None,
        return_features=False,
        attention_mask=None,
        position_ids=None,
    ):
        return self.forward_with_embeddings(
            self.embed_state(log_state),
            time.float(),
            self_condition=self_condition,
            return_features=return_features,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

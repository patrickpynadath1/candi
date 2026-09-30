"""CANDI backbone with position-aligned feedback from its final block."""

import torch
from torch import nn

from .dit_cont import ContDIT
from .registry import models


@models.register("looped_cont_dit")
class LoopedContDIT(ContDIT):
    def __init__(self, config, vocab_size):
        super().__init__(config, vocab_size)
        self.feedback_norm = nn.LayerNorm(self.dim, elementwise_affine=False)
        self.feedback_projection = nn.Linear(self.dim, self.dim, bias=False)

    def _condition_hidden(self, hidden, previous_hidden):
        if previous_hidden is None:
            previous_hidden = torch.zeros_like(hidden)
        if previous_hidden.shape != hidden.shape:
            raise ValueError("previous_hidden must match [batch, length, hidden_size]")
        # Evaluate the projection even on the first pass so all parameters
        # participate in the graph for distributed training with small batches.
        feedback = self.feedback_norm(previous_hidden.to(hidden))
        return hidden + self.feedback_projection(feedback)

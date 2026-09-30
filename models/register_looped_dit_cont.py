"""CANDI with a small set of recurrent attention registers."""

import torch
from torch import nn

from .dit_cont import ContDIT
from .registry import models


@models.register("register_looped_cont_dit")
class RegisterLoopedContDIT(ContDIT):
    def __init__(self, config, vocab_size):
        super().__init__(config, vocab_size)
        count = self.config.algo.num_registers
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("algo.num_registers must be a positive integer")
        self.num_registers = count
        self.register_embedding = nn.Parameter(torch.empty(count, self.dim))
        nn.init.normal_(self.register_embedding, std=self.dim**-0.5)
        self.feedback_norm = nn.LayerNorm(self.dim, elementwise_affine=False)
        self.feedback_projection = nn.Linear(self.dim, self.dim, bias=False)

    def _condition_hidden(self, hidden, previous_hidden):
        registers = self.register_embedding.to(hidden).unsqueeze(0).expand(
            hidden.shape[0], -1, -1
        )
        if previous_hidden is None:
            previous_hidden = torch.zeros_like(registers)
        if previous_hidden.shape != registers.shape:
            raise ValueError(
                "previous_hidden must match [batch, num_registers, hidden_size]"
            )
        registers = registers + self.feedback_projection(
            self.feedback_norm(previous_hidden.to(hidden))
        )
        return torch.cat((hidden, registers), dim=1)

    def forward(self, *args, ret_embedding=False, **kwargs):
        logits, hidden = super().forward(*args, ret_embedding=True, **kwargs)
        # The shared backbone attends to registers but vocabulary predictions
        # and recurrence expose disjoint position sets.
        data_logits = logits[:, :-self.num_registers]
        register_hidden = hidden[:, -self.num_registers:].clone()
        if ret_embedding:
            return data_logits, register_hidden
        return data_logits

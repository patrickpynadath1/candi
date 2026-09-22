"""CANDI recurrence through probability-weighted input embeddings."""

import torch.nn.functional as F

from .dit_cont import ContDIT
from .registry import models


@models.register("softmax_looped_cont_dit")
class SoftmaxLoopedContDIT(ContDIT):
    def _condition_hidden(self, hidden, previous_hidden):
        if previous_hidden is None:
            return hidden
        table = self.vocab_embed.embedding
        if self.reserve_mask_embedding:
            table = table[:-1]
        if previous_hidden.shape != (*hidden.shape[:2], table.shape[0]):
            raise ValueError("feedback must match [batch, length, data_vocab_size]")
        feedback = previous_hidden.to(table) @ table
        if not self.legacy_architecture:
            # Equivalent to projecting (preconditioned input + feedback),
            # without adding the input projection's bias a second time.
            feedback = F.linear(feedback, self.input_projection.weight)
        return hidden + feedback.to(hidden)

    def forward(self, *args, ret_embedding=False, **kwargs):
        logits = super().forward(*args, ret_embedding=False, **kwargs)
        if ret_embedding:
            # The recurrent payload is a prediction, not a transformer state.
            return logits, logits.float().softmax(dim=-1)
        return logits

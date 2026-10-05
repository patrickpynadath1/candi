"""Intermediate token prediction and hidden-space feedback execution."""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy

import torch
from torch import nn

from core.models.components import BlockExecutionOutput


class TokenFeedbackBlocks(nn.Module):
    """Run one auxiliary extractor at a fixed depth, then continue the backbone."""

    uses_intermediate_feedback = True

    def __init__(self, template, depth, hidden_size, *, gate="scalar", detach=False):
        super().__init__()
        if gate not in {"scalar", "channel"}:
            raise ValueError("token feedback gate must be scalar or channel")
        self.extractor = deepcopy(template)
        self.depth = int(depth)
        self.detach = bool(detach)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        shape = () if gate == "scalar" else (hidden_size,)
        self.gate = nn.Parameter(torch.zeros(shape))
        self._capture = ContextVar("token_feedback_capture", default=None)

    @contextmanager
    def capture(self):
        values = []
        token = self._capture.set(values)
        try:
            yield values
        finally:
            self._capture.reset(token)

    def forward(
        self,
        hidden,
        blocks,
        rotary,
        *,
        condition=None,
        attention_mask=None,
        project=None,
        feedback_embedding=None,
    ):
        if not 0 <= self.depth < len(blocks):
            raise ValueError(
                f"token feedback depth {self.depth} must be in [0, {len(blocks) - 1}]"
            )
        if project is None or feedback_embedding is None:
            raise TypeError(
                "token feedback requires an explicit token output head and token embedding table"
            )
        auxiliary_logits = None
        for index, block in enumerate(blocks):
            hidden = block(hidden, rotary, condition, attention_mask)
            if index == self.depth:
                extracted = self.extractor(hidden, rotary, condition, attention_mask)
                auxiliary_logits = project(extracted, condition)
                if auxiliary_logits.shape[-1] != feedback_embedding.shape[0]:
                    raise TypeError(
                        "token feedback output vocabulary must match the token embedding table; "
                        "score/velocity outputs need an explicit token readout"
                    )
                weights = auxiliary_logits.float().softmax(-1)
                if self.detach:
                    weights = weights.detach()
                feedback = weights @ feedback_embedding.float()
                feedback = self.norm(feedback).to(hidden.dtype)
                hidden = hidden + feedback * self.gate.to(hidden.dtype)
        capture = self._capture.get()
        if capture is not None and torch.is_grad_enabled():
            capture.append(auxiliary_logits)
        return BlockExecutionOutput(hidden, auxiliary_logits)

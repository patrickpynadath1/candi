"""Algorithm-level auxiliary clean-token objective for token feedback."""

import torch
from torch import nn
from torch.nn import functional as F
import weakref

from .model import TokenFeedbackBlocks


def _backbone(model):
    while hasattr(model, "model"):
        model = model.model
    return getattr(model, "backbone", model)


class RepresentationFeedback(nn.Module):
    """Map clean representation codewords through the model's existing input map."""

    def __init__(self, representation, projection, embedding=None):
        super().__init__()
        self.representation = representation
        self.projection = projection
        self.embedding = embedding

    def forward(self):
        table = self.representation.encode(
            torch.arange(
                self.representation.spec.vocab_size,
                device=self.projection.weight.device,
            )
        )
        with torch.autocast(table.device.type, enabled=False):
            table = table.float()
            if self.embedding is not None:
                table = table @ F.normalize(self.embedding.float(), dim=-1)
            return self.projection(table)


class TokenFeedback(nn.Module):
    def __init__(
        self, algorithm, *, depth, loss_weight=1.0, gate="scalar", detach=False
    ):
        super().__init__()
        if loss_weight < 0:
            raise ValueError("token feedback loss weight must be nonnegative")
        self.algorithm = algorithm
        model = _backbone(algorithm.model)
        blocks = getattr(model, "blocks", None)
        table = (
            model.feedback_embedding_table()
            if hasattr(model, "feedback_embedding_table")
            else None
        )
        representation = getattr(algorithm, "representation", None)
        if representation is None and hasattr(algorithm, "views"):
            representation = getattr(algorithm.views[0], "representation", None)
        if (
            table is None
            and representation is not None
            and hasattr(model, "input_projection")
        ):
            model.feedback_table_adapter = RepresentationFeedback(
                representation,
                model.input_projection,
                getattr(model, "input_embedding", None),
            )
            table = model.feedback_embedding_table()
        if blocks is None or not blocks or table is None:
            raise TypeError(
                "token feedback requires a transformer with a compatible token embedding table"
            )
        if not 0 <= depth < len(blocks):
            raise ValueError(f"token feedback depth must be in [0, {len(blocks) - 1}]")
        hidden_size = table.shape[1]
        model.block_execution = TokenFeedbackBlocks(
            blocks[depth], depth, hidden_size, gate=gate, detach=detach
        )
        self._feedback_ref = weakref.ref(model.block_execution)
        self.loss_weight = float(loss_weight)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.algorithm, name)

    @property
    def model(self):
        return self.algorithm.model

    @property
    def feedback(self):
        return self._feedback_ref()

    def loss(self, tokens, loss_mask, *args, **kwargs):
        training = bool(kwargs.get("training", False))
        if not training or self.loss_weight == 0:
            return self.algorithm.loss(tokens, loss_mask, *args, **kwargs)
        with self.feedback.capture() as captured:
            result = self.algorithm.loss(tokens, loss_mask, *args, **kwargs)
        if len(captured) != 1:
            raise RuntimeError(
                "token feedback training expected exactly one gradient-enabled model pass"
            )
        logits = captured[0]
        if logits.shape[1] != tokens.shape[1]:
            logits = logits[:, -tokens.shape[1] :]
        weights = loss_mask.to(logits)
        count = weights.sum()
        if count <= 0:
            raise ValueError("loss mask must include at least one target")
        per_token = F.cross_entropy(logits.transpose(1, 2), tokens, reduction="none")
        auxiliary_sum = (per_token * weights).sum()
        auxiliary_loss = auxiliary_sum / count
        result["main_loss"] = result["loss"]
        result["token_feedback_loss"] = auxiliary_loss
        result["loss"] = result["loss"] + self.loss_weight * auxiliary_loss
        result["loss_sum"] = result["loss"].detach() * result["num_tokens"]
        return result

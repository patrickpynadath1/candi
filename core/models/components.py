"""Interchangeable, parameter-free model execution components.

Learned tensors stay on the public model so existing checkpoint paths, weight
sharing, and optimizer traversal remain unchanged. Components receive those
modules explicitly; they never inspect a representation or predictor.
"""

import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from contextlib import nullcontext


@dataclass
class BlockExecutionOutput:
    hidden: torch.Tensor
    auxiliary_logits: torch.Tensor | None = None


class TokenInput(nn.Module):
    def forward(self, tokens, embedding):
        return embedding(tokens)


class SimplexExpectationInput(nn.Module):
    def forward(self, state, embedding):
        return state.exp().to(embedding.embedding.dtype) @ embedding.embedding


class SimplexArgmaxInput(nn.Module):
    def forward(self, state, embedding):
        return embedding(state.argmax(-1)).masked_fill(
            ~torch.isfinite(state).any(-1)[..., None], 0
        )


class LinearStateInput(nn.Module):
    def forward(self, state, projection, *, embedding=None, precondition=None):
        hidden = state.float()
        if precondition is not None:
            hidden = precondition(hidden)
        return projection(hidden)


class EmbeddedStateInput(nn.Module):
    def forward(self, state, projection, *, embedding, precondition=None):
        hidden = state.float() @ F.normalize(embedding.float(), dim=-1)
        if precondition is not None:
            hidden = precondition(hidden)
        return projection(hidden)


class TimeConditioning(nn.Module):
    def forward(self, coordinate, embedder):
        return F.silu(embedder(coordinate))


class LayerOutput(nn.Module):
    def forward(self, hidden, condition, *, layer):
        return layer(hidden, condition)


class EmbeddingOutput(nn.Module):
    def forward(self, hidden, condition, *, projection, embedding, bias):
        return projection(hidden) @ embedding.t() + bias


class TemperatureOutput(LayerOutput):
    def forward(self, hidden, condition, *, layer):
        output = super().forward(hidden, condition, layer=layer)
        logits, raw_temp = output[..., :-1], output[..., -1:]
        return logits / (1.000001 + torch.tanh(raw_temp) * 0.999999)


class DiffusionBlocks(nn.Module):
    def __init__(self, activation_checkpointing=False):
        super().__init__()
        self.activation_checkpointing = activation_checkpointing

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
        del project, feedback_embedding
        for block in blocks:
            if self.activation_checkpointing and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

                # Legacy TorchScript residual helpers wrap checkpoint's internal
                # early-stop exception. Recompute the complete block instead.
                # Compiled execution inlines these helpers, so it can retain
                # checkpoint's early-stop optimization without Python context.
                context = (
                    nullcontext()
                    if torch.compiler.is_compiling()
                    else set_checkpoint_early_stop(False)
                )
                with context:
                    hidden = checkpoint(
                        block,
                        hidden,
                        rotary,
                        condition,
                        attention_mask,
                        use_reentrant=False,
                    )
            else:
                hidden = block(hidden, rotary, condition, attention_mask)
        return hidden


class CausalBlocks(nn.Module):
    def forward(self, hidden, blocks, rotary, *, cache=None, query_rotary=None):
        for index, block in enumerate(blocks):
            hidden = block(
                hidden,
                rotary,
                None if cache is None else cache.layers[index],
                query_rotary,
            )
        return hidden

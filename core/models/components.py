"""Interchangeable, parameter-free model execution components.

Learned tensors stay on the public model so existing checkpoint paths, weight
sharing, and optimizer traversal remain unchanged. Components receive those
modules explicitly; they never inspect a representation or predictor.
"""

import torch
from torch import nn
from dataclasses import dataclass
from contextlib import nullcontext


@dataclass
class BlockExecutionOutput:
    hidden: torch.Tensor
    auxiliary_logits: torch.Tensor | None = None


class TokenInput(nn.Module):
    def forward(self, tokens, embedding):
        return embedding(tokens)


class LayerOutput(nn.Module):
    def forward(self, hidden, condition, *, layer):
        return layer(hidden, condition)


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

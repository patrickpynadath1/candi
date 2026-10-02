"""Interchangeable, parameter-free model execution components.

Learned tensors stay on the public model so existing checkpoint paths, weight
sharing, and optimizer traversal remain unchanged. Components receive those
modules explicitly; they never inspect a representation or predictor.
"""

import torch
from torch import nn
from torch.nn import functional as F


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
    def forward(self, hidden, blocks, rotary, *, condition=None,
                attention_mask=None):
        for block in blocks:
            hidden = block(hidden, rotary, condition, attention_mask)
        return hidden


class CausalBlocks(nn.Module):
    def forward(self, hidden, blocks, rotary, *, cache=None, query_rotary=None):
        for index, block in enumerate(blocks):
            hidden = block(hidden, rotary,
                           None if cache is None else cache.layers[index],
                           query_rotary)
        return hidden

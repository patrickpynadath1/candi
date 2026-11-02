"""Query-dependent depth routing over an encoder's actual attention K/V.

No corruption assumptions or extra context projections. Context tensors have
shape [batch, position, encoder_depth, head, head_dim]. Keys already include
position encoding. Depth selection precedes joint context/noisy attention.
"""

import math

import torch
from torch import nn


class EncoderDepthAttention(nn.Module):
    def __init__(self, query_chunk_size=32):
        super().__init__()
        if type(query_chunk_size) is not int or query_chunk_size < 1:
            raise ValueError("query_chunk_size must be a positive integer")
        self.query_chunk_size = query_chunk_size

    def forward(self, query, key, value, context, attention_mask=None):
        with torch.autocast(query.device.type, enabled=False):
            return self._attend(query, key, value, context, attention_mask)

    def _attend(self, query, key, value, context, attention_mask):
        context_key, context_value = context
        outputs = []
        scale = 1 / math.sqrt(query.shape[-1])
        # Query tiling bounds each intermediate score tensor. Autograd still
        # retains tiles during training; checkpointing is recommended.
        for start in range(0, query.shape[1], self.query_chunk_size):
            q = query[:, start : start + self.query_chunk_size].float()
            scores = torch.einsum("bqhd,bclhd->bhqcl", q, context_key.float()) * scale
            depth_weights = scores.softmax(dim=-1)
            context_scores = (depth_weights * scores).sum(dim=-1)
            noisy_scores = torch.einsum("bqhd,bkhd->bhqk", q, key.float()) * scale
            scores = torch.cat((context_scores, noisy_scores), dim=-1)
            if attention_mask is not None:
                mask = attention_mask
                if mask.shape[-2] != 1:
                    mask = mask[..., start : start + q.shape[1], :]
                if mask.dtype == torch.bool:
                    scores = scores.masked_fill(~mask, -torch.inf)
                else:
                    scores = scores + mask
            # Like SDPA, an entirely masked query produces zero output.
            all_masked = torch.isneginf(scores).all(dim=-1, keepdim=True)
            weights = scores.masked_fill(all_masked, 0).softmax(dim=-1)
            weights = weights.masked_fill(all_masked, 0)
            context_weights = weights[..., : context_key.shape[1]]
            noisy_weights = weights[..., context_key.shape[1] :]
            output = torch.einsum(
                "bhqcl,bclhd->bqhd",
                context_weights.unsqueeze(-1) * depth_weights,
                context_value.float(),
            )
            output = output + torch.einsum(
                "bhqk,bkhd->bqhd", noisy_weights, value.float()
            )
            outputs.append(output.to(query.dtype).flatten(-2))
        return torch.cat(outputs, dim=1)

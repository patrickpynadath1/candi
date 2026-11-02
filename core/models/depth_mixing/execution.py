"""Depth routing around the shared transformer block computation."""

import torch
from torch import nn


class DepthMixingBlocks(nn.Module):
    """Select attention inputs from history without changing token attention.

    ``states`` implements MUDD-style whole-block history, including the final
    residual readout. ``contributions`` implements DCA's history of residual
    branch updates, followed by a final learned GRN readout.
    """

    def __init__(self, dim, n_blocks, mixer, history="states"):
        super().__init__()
        if history not in {"states", "contributions"}:
            raise ValueError("history must be states or contributions")
        self.history = history
        start = 2 if history == "states" else 1
        self.mixers = nn.ModuleList(
            mixer(dim=dim, num_sources=s) for s in range(start, n_blocks + 2)
        )

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
        expected = len(blocks) + (self.history == "contributions")
        if expected != len(self.mixers):
            raise ValueError("depth mixer count must match the backbone blocks")
        history = [hidden]
        streams = hidden.unsqueeze(2).expand(-1, -1, 4, -1)
        for block, mixer in zip(blocks, self.mixers):
            if self.history == "contributions":
                streams = mixer(torch.stack(history, dim=2))
            hidden, contribution = block.forward_streams(
                streams, rotary, condition, attention_mask
            )
            history.append(contribution if self.history == "contributions" else hidden)
            if self.history == "states":
                streams = mixer(torch.stack(history, dim=2))
        if self.history == "states":
            return streams[:, :, 3]
        return self.mixers[-1](torch.stack(history, dim=2))[:, :, 3]

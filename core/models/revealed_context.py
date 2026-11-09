"""Continuous input plus revealed clean context, independent of self-conditioning."""
import torch
from torch import nn


class RevealedContextInput(nn.Module):
    state_kind = "views"

    def __init__(self, views, continuous_input, hidden_size, context_scale=1.0):
        super().__init__()
        self.continuous = continuous_input
        self.views = nn.ModuleList(views)
        dimension = views[0].representation.spec.dimension
        self.context_projection = nn.Linear(dimension, hidden_size, bias=False)
        self.context_scale = dimension**0.5 if context_scale == "unit_embedding" else float(context_scale)

    def feedback(self):
        return self.views[0].representation.codebook()

    def forward(self, states, points):
        tokens, known = self.views[1].process.observed_tokens(states[1])
        clean = self.views[0].representation.encode(torch.where(known, tokens, 0))
        context = torch.where(known[..., None], clean, 0.)
        return self.continuous(states[0], points[0]) + self.context_projection(context*self.context_scale)

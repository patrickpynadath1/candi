"""Model input slot: a continuous view plus the clean context its token view reveals.

Select it as ``denoising_task/model/input@model.input: revealed_context``. The slot
holds two views (a continuous content view, then an absorbing token view): the
continuous slot embeds the noisy state, and every position the token view has
revealed adds a projection of its clean embedding. It is not posterior
self-conditioning; it reads only the current noisy states.
"""
from collections.abc import Mapping

import torch
from torch import nn


class RevealedContextInput(nn.Module):
    state_kind = "views"

    def __init__(self, views, continuous, hidden_size, state_dim=None, context_scale=1.0):
        super().__init__()
        from .modular import _build

        if isinstance(views, Mapping):
            views = list(views.values())
        if len(views) != 2:
            raise ValueError("revealed context needs a continuous view and a token view")
        if not callable(getattr(views[1].process, "observed_tokens", None)):
            raise TypeError("the second view must reveal tokens (an absorbing process)")
        dimension = views[0].representation.spec.dimension
        if state_dim is not None and state_dim != dimension:
            raise ValueError("model state_dim must match the continuous view's dimension")
        # A partial (from config) or a built module.
        self.continuous = (
            continuous
            if isinstance(continuous, nn.Module)
            else _build(continuous, state_dim=dimension, hidden_size=hidden_size)
        )
        self.views = nn.ModuleList(views)
        self.context_projection = nn.Linear(dimension, hidden_size, bias=False)
        self.context_scale = (
            dimension**0.5 if context_scale == "unit_embedding" else float(context_scale)
        )

    def feedback(self):
        return self.views[0].representation.codebook()

    def forward(self, states, points):
        tokens, known = self.views[1].process.observed_tokens(states[1])
        clean = self.views[0].representation.encode(torch.where(known, tokens, 0))
        context = torch.where(known[..., None], clean, 0.)
        return self.continuous(states[0], points[0]) + self.context_projection(context*self.context_scale)

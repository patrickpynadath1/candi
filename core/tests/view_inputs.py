"""Call one-view token networks from tests written for the former DiT form."""

import torch

from core.modifiers.hybrid.views import ViewPoint


def token_call(model, tokens, conditioning, **options):
    """``model(tokens, conditioning)`` for a one-view token network.

    The former DiT took MDLM's time input directly (zero, or -log keep); a view
    network reads it from the view's keep rate, so keep = exp(-conditioning).
    """
    conditioning = torch.as_tensor(conditioning, device=tokens.device).float()
    keep = torch.exp(-conditioning)
    point = ViewPoint(conditioning, (keep, torch.zeros_like(keep)))
    return model((tokens,), (point,), conditioning, **options)


def keep_points(keep):
    """One-view points for a keep-rate (MDLM-style) recipe at the given keep."""
    return (ViewPoint(1 - keep, (keep, torch.zeros_like(keep))),)


def view_prediction(endpoint, state, keep, **options):
    """``endpoint(state, keep)`` for a one-view recipe's prediction endpoint."""
    return endpoint((state,), keep_points(keep), 1 - keep, **options)

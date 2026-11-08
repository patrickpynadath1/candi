"""Small ModularDiT networks for tests, matching the configs/model presets."""

from functools import partial

import torch

from core.models import modular
from core.modifiers.hybrid.views import ViewPoint
from core.noise_schedules.gaussian_path import GaussianPath


def _self_conditioning(enabled, source, **options):
    if callable(enabled):  # An explicit slot (e.g. one listed with enabled=False).
        return enabled
    if not enabled:
        return None
    return partial(modular.SelfConditioning, source=source, **options)


def token(
    vocab_size,
    hidden_size=16,
    cond_dim=8,
    n_blocks=1,
    n_heads=2,
    *,
    dropout=0.0,
    adaLN=False,
    self_conditioning=False,
    source="posterior_mean",
    views=None,
    fusion=None,
    coordinate="time",
    **options,
):
    """configs/model/dit.yaml: token (or views) input, AdaLN head."""
    return modular.ModularDiT(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        cond_dim=cond_dim,
        n_blocks=n_blocks,
        n_heads=n_heads,
        dropout=dropout,
        input=None if views is not None else partial(modular.TokenInput),
        views=views,
        fusion=fusion,
        time=partial(modular.TimeConditioning, coordinate=coordinate, adaLN=adaLN),
        output=partial(modular.AdaLNHead, adaLN=adaLN),
        block_adaLN=adaLN,
        self_conditioning=_self_conditioning(self_conditioning, source),
        **options,
    )


def continuous(
    vocab_size,
    state_dim=None,
    hidden_size=16,
    cond_dim=8,
    n_blocks=1,
    n_heads=2,
    *,
    dropout=0.0,
    input="embedded_projection",
    preconditioning="gaussian",
    bias=False,
    coordinate="log_one_minus_time",
    epsilon=1e-6,
    time_input=False,
    output="embedding",
    output_dim=None,
    self_conditioning=False,
    source="posterior_mean",
):
    """configs/model/continuous_dit.yaml with its slot options."""
    state_dim = vocab_size if state_dim is None else state_dim
    input_slot = (
        partial(modular.LinearInput, bias=bias, preconditioning=preconditioning)
        if input == "linear"
        else partial(modular.EmbeddedProjectionInput, preconditioning=preconditioning)
    )
    output_slot = (
        partial(modular.AdaLNHead)
        if output == "adaln"
        else partial(modular.EmbeddingHead, output_dim=output_dim)
    )
    return modular.ModularDiT(
        vocab_size=vocab_size,
        state_dim=state_dim,
        hidden_size=hidden_size,
        cond_dim=cond_dim,
        n_blocks=n_blocks,
        n_heads=n_heads,
        dropout=dropout,
        autocast_cache=False,
        input=input_slot,
        time=partial(
            modular.TimeConditioning,
            coordinate=coordinate,
            epsilon=epsilon,
            input_projection=time_input,
        ),
        output=output_slot,
        self_conditioning=_self_conditioning(self_conditioning, source, space="state"),
    )


def sphere(
    vocab_size,
    state_dim,
    hidden_size=16,
    cond_dim=8,
    n_blocks=1,
    n_heads=2,
    *,
    dropout=0.0,
    temperature=False,
    self_conditioning=False,
    source="posterior_mean",
):
    """configs/model/sphere_dit.yaml: linear input, AdaLN head (optional temperature)."""
    return modular.ModularDiT(
        vocab_size=vocab_size,
        state_dim=state_dim,
        hidden_size=hidden_size,
        cond_dim=cond_dim,
        n_blocks=n_blocks,
        n_heads=n_heads,
        dropout=dropout,
        input=partial(modular.LinearInput, bias=True),
        time=partial(modular.TimeConditioning, coordinate="time"),
        output=partial(modular.AdaLNHead, temperature=temperature),
        self_conditioning=_self_conditioning(self_conditioning, source, space="state"),
    )


def path_call(model, state, time, sigma, alpha, **options):
    """A one-state network at global ``time`` on a Gaussian path (alpha, sigma)."""
    zeros = torch.zeros_like(alpha)
    point = ViewPoint(time, GaussianPath(alpha, sigma, zeros, zeros))
    return model((state,), (point,), time, **options)

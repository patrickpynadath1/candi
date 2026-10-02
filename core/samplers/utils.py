"""Shared generation setup for samplers."""

import torch
from core.parameterizations.posterior import PosteriorParameterization


def initialize_state(
    algo, prompt_tokens, prompt_mask, predictor_type=PosteriorParameterization
):
    if not isinstance(algo.predictor, predictor_type):
        raise TypeError(f"sampler requires {predictor_type.__name__}")
    if prompt_tokens.ndim != 2 or prompt_tokens.shape != prompt_mask.shape:
        raise ValueError("prompt tensors must have matching [batch, length] shapes")
    if prompt_tokens.shape[1] != algo.length or len(prompt_tokens) == 0:
        raise ValueError("invalid prompt batch or sequence length")
    if torch.any(prompt_tokens[prompt_mask.bool()] == algo.process.mask_index):
        raise ValueError("fixed prompt cannot contain the absorbing mask")
    return algo.process.prior(
        prompt_tokens.shape, prompt_tokens.device, prompt_tokens, prompt_mask
    )


def sample_categorical(probabilities):
    """Exponential-race categorical draw preserving the legacy RNG formula."""
    exponential = 1e-10 - (torch.rand_like(probabilities) + 1e-10).log()
    return (probabilities / exponential).argmax(-1)

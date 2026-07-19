from __future__ import annotations

import torch

from .base import Sampler, SamplingRequest, SamplingResult


@torch.no_grad()
def run_sampler(
    sampler: Sampler, model, request: SamplingRequest
) -> SamplingResult:
    if request.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if request.num_steps is not None and request.num_steps <= 0:
        raise ValueError("num_steps must be positive when provided")
    if (request.prompt_tokens is None) != (request.prompt_mask is None):
        raise ValueError("prompt_tokens and prompt_mask must be provided together")
    if request.prompt_tokens is not None:
        if request.prompt_tokens.shape != request.prompt_mask.shape:
            raise ValueError("prompt tokens and mask must have matching shapes")
        if request.prompt_tokens.ndim != 2:
            raise ValueError("prompt tensors must have shape [batch, sequence]")
        if request.prompt_tokens.shape[0] != request.num_samples:
            raise ValueError("prompt batch size must equal num_samples")
    sampler.validate_model(model)
    return sampler.sample(model, request)

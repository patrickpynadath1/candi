from __future__ import annotations

import torch

from .base import Sampler, SamplingRequest, SamplingResult


@torch.no_grad()
def run_sampler(
    sampler: Sampler, model, request: SamplingRequest
) -> SamplingResult:
    sampler.validate_model(model)
    return sampler.sample(model, request)

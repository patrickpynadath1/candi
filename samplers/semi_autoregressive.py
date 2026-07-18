from __future__ import annotations

import torch

from .base import Sampler, SamplerSpec, SamplingResult
from .registry import sampler_registry


class SemiAutoregressiveSampler(Sampler):
    spec = SamplerSpec(
        name="semi_ar",
        compatible_families=frozenset(
            {"absorbing_diffusion", "uniform_diffusion"}
        ),
        required_capabilities=frozenset(
            {"discrete_prior", "ancestral_transition", "zero_time_denoise"}
        ),
    )

    def sample(self, model, request):
        stride = request.options["stride_length"]
        num_strides = request.options["num_strides"]
        dt = request.options.get("dt", 0.001)
        ones = torch.ones(request.num_samples, dtype=model.dtype, device=model.device)
        num_steps = int(1 / dt)
        nfe = 0
        chunks = []
        target = None
        for _ in range(num_strides + 1):
            cache = None
            x = model.prior_sample(request.num_samples, model.num_tokens)
            if target is not None:
                x[:, :-stride] = target
            for i in range(num_steps + 1):
                cache, x_next = model.sample_ancestral_transition(
                    x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=cache
                )
                if not torch.allclose(x_next, x) or model.time_conditioning:
                    cache = None
                    nfe += 1
                x = x_next
            x = model.sample_zero_time_denoise(x, ones)
            chunks.append(x[:, :stride])
            target = x[:, stride:]
        chunks.append(target)
        samples = torch.cat(chunks, dim=1)
        return SamplingResult(samples, {"nfe": nfe, "chunks": chunks})


@sampler_registry.register("semi_ar")
def _semi_ar(_config):
    return SemiAutoregressiveSampler()

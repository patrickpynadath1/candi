"""Euler integration of the FLM probability-flow ODE."""

import torch

from .base import Sampler, SamplerSpec, SamplingResult
from .registry import sampler_registry
from .utils import preserve_prompt


def flm_euler_step(z, x1, gamma_t, gamma_delta, eps=1e-5):
    while gamma_t.ndim < z.ndim:
        gamma_t = gamma_t.unsqueeze(-1)
    while gamma_delta.ndim < z.ndim:
        gamma_delta = gamma_delta.unsqueeze(-1)
    return z + gamma_delta * (x1 - z) / (1 - gamma_t).clamp_min(eps)


class FLMEulerSampler(Sampler):
    spec = SamplerSpec(
        "flm_euler",
        frozenset({"flm"}),
        frozenset({"flm_prior", "flm_time_map", "flm_logits"}),
    )

    def __init__(self, use_float64=False, temperature=1.0):
        self.use_float64 = use_float64
        self.temperature = temperature

    def sample(self, model, request):
        steps = request.num_steps or model.config.sampling.steps
        x = model.prior_sample(request.num_samples, model.num_tokens)
        times = torch.linspace(0, 1, steps + 1, device=model.device)
        prompt_mask = (
            request.prompt_mask.bool() if request.prompt_mask is not None else None
        )
        if prompt_mask is not None:
            prompt = torch.nn.functional.one_hot(
                request.prompt_tokens, model.vocab_size
            ).to(x)
            x = torch.where(prompt_mask.unsqueeze(-1), prompt, x)
        for i in range(steps):
            t = times[i].expand(request.num_samples)
            tn = times[i + 1].expand_as(t)
            gt = model.sampling_gamma(t)
            gn = model.sampling_gamma(tn)
            logp = model.sampling_logits(x, t) / self.temperature
            if self.use_float64:
                x = x.double()
                logp = logp.double()
                gt = gt.double()
                gn = gn.double()
            if i < steps - 1:
                x = flm_euler_step(x, logp.softmax(-1), gt, gn - gt)
            if prompt_mask is not None:
                x = torch.where(prompt_mask.unsqueeze(-1), prompt.to(x), x)
        return SamplingResult(preserve_prompt(logp.argmax(-1), request), {"nfe": steps})


@sampler_registry.register("flm_euler")
def _create(config):
    return FLMEulerSampler(
        getattr(config.sampling, "use_float64", False),
        getattr(config.sampling, "temperature", 1.0),
    )

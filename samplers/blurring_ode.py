"""Probability-flow Euler sampler for heat-kernel blurring diffusion."""

import torch

from .base import Sampler, SamplerSpec, SamplingResult
from .registry import sampler_registry


class BlurringODESampler(Sampler):
    spec = SamplerSpec(
        name="blurring_ode",
        compatible_families=frozenset({"blurring_diffusion"}),
        required_capabilities=frozenset(
            {"blurring_prior", "blurring_transition"}
        ),
    )

    @torch.no_grad()
    def sample(self, model, request):
        steps = request.num_steps or model.config.sampling.steps
        times = torch.linspace(1.0, 0.0, steps + 1, device=model.device)
        state = model.prior_sample(request.num_samples, model.num_tokens)
        if request.prompt_mask is None:
            active = torch.ones(
                state.shape[:2], device=state.device, dtype=torch.bool
            )
        else:
            if request.prompt_tokens is None:
                raise ValueError("prompt_tokens and prompt_mask must be provided together")
            active = ~request.prompt_mask.bool()
            prompt = model.clean_prompt_embedding(request.prompt_tokens).to(state)
            state = torch.where(request.prompt_mask.bool().unsqueeze(-1), prompt, state)

        probabilities = None
        for index in range(steps):
            state, probabilities = model.sample_step(
                state, times[index], times[index + 1], active
            )
        final_time = torch.zeros(
            state.shape[0], device=state.device, dtype=state.dtype
        )
        probabilities = model.forward(state, final_time).exp()
        tokens = probabilities.argmax(-1)
        if request.prompt_mask is not None:
            tokens = torch.where(
                request.prompt_mask.bool(), request.prompt_tokens, tokens
            )
        return SamplingResult(tokens, {"nfe": steps})


@sampler_registry.register("blurring_ode")
def _create(_config):
    return BlurringODESampler()

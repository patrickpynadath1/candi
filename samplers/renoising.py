from __future__ import annotations

import torch

from .base import Sampler, SamplerSpec, SamplingResult
from .entropy_utils import sample_probabilities
from .registry import sampler_registry
from .utils import continuous_time_grid, preserve_prompt


class RenoisingSampler(Sampler):
    """Iterative clean prediction followed by independent forward re-corruption."""

    spec = SamplerSpec(
        name="renoising",
        compatible_families=frozenset({"candi"}),
        required_capabilities=frozenset({
            "continuous_prior",
            "continuous_schedule",
            "candi_discrete_schedule",
            "candi_clean_distribution",
            "candi_forward_corruption",
        }),
    )

    def sample(self, model, request):
        steps, times = continuous_time_grid(model, request)
        state = model.prior_sample(request.num_samples, model.num_tokens)
        state, reveal_mask = model.sampling_prompt_state(
            state, request.prompt_tokens, request.prompt_mask
        )
        continuous_noise = model.sampling_continuous_schedule(times, steps)
        discrete_noise = model.sampling_discrete_schedule(times)
        generated_mask = torch.ones(
            state.shape[:2], device=state.device, dtype=torch.bool
        )
        if request.prompt_mask is not None:
            generated_mask &= ~request.prompt_mask.bool()

        clean_tokens = None
        for index in range(steps):
            clean_probs = model.sample_candi_clean_distribution(
                state,
                discrete_noise[index],
                continuous_noise[index],
                reveal_mask,
            )
            clean_tokens = sample_probabilities(clean_probs)
            clean_tokens = preserve_prompt(clean_tokens, request)
            if index + 1 < steps:
                state, reveal_mask = model.sample_candi_forward_corruption(
                    clean_tokens,
                    discrete_noise[index + 1],
                    continuous_noise[index + 1],
                    generated_mask,
                )

        return SamplingResult(clean_tokens, {
            "nfe": steps,
            "sampler": "renoising",
        })


@sampler_registry.register("renoising")
def _create(_config):
    return RenoisingSampler()

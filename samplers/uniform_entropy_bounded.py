from __future__ import annotations

import torch

from .base import Sampler, SamplerSpec, SamplingResult
from .entropy_utils import predictive_entropy, sample_probabilities, select_entropy_bounded
from .registry import sampler_registry
from .utils import apply_token_offset, diffusion_time_grid, inject_token_prompt, preserve_prompt


class UniformEntropyBoundedSampler(Sampler):
    """Freeze entropy-bounded predictions while rejected sites keep denoising."""

    spec = SamplerSpec(
        name="uniform_entropy_bounded",
        compatible_families=frozenset({"uniform_diffusion"}),
        required_capabilities=frozenset({
            "discrete_prior", "noise_schedule", "uniform_entropy_transition",
        }),
    )

    def __init__(self, gamma, greedy_clean_estimate=False):
        if gamma < 0:
            raise ValueError("sampling.entropy_bound must be nonnegative")
        self.gamma = float(gamma)
        self.greedy_clean_estimate = bool(greedy_clean_estimate)

    def sample(self, model, request):
        steps, times, dt = diffusion_time_grid(model, request)
        x = inject_token_prompt(
            model.prior_sample(request.num_samples, model.num_tokens), request
        )
        accepted = (
            torch.zeros_like(x, dtype=torch.bool)
            if request.prompt_mask is None
            else request.prompt_mask.bool().clone()
        )
        accepted_per_step = []
        nfe_per_sample = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        for i in range(steps):
            active = ~accepted
            if not active.any():
                break
            t = times[i] * torch.ones(x.shape[0], 1, device=model.device)
            log_probs, rejected_next = model.sample_uniform_entropy_transition(
                x=x, t=t, dt=dt, frozen_mask=accepted,
                greedy_clean_estimate=self.greedy_clean_estimate,
            )
            probs = log_probs.float().exp()
            entropy = predictive_entropy(probs)
            newly_accepted, counts = select_entropy_bounded(
                entropy, active, self.gamma
            )
            clean_samples = (
                probs.argmax(dim=-1)
                if self.greedy_clean_estimate
                else sample_probabilities(probs)
            )
            # Rejected sites continue the ordinary reverse trajectory. Accepted
            # sites become clean tokens and remain frozen on later steps.
            x = torch.where(newly_accepted, clean_samples, rejected_next)
            accepted |= newly_accepted
            x = preserve_prompt(x, request)
            accepted_per_step.append(counts)
            newly_complete = nfe_per_sample.eq(0) & accepted.all(dim=-1)
            nfe_per_sample[newly_complete] = i + 1

        if not accepted.all():
            raise RuntimeError(
                "uniform entropy-bounded sampling exhausted sampling.steps "
                "before accepting every position"
            )
        nfe = len(accepted_per_step)
        return SamplingResult(apply_token_offset(model, x), {
            "nfe": nfe,
            "nfe_per_sample": nfe_per_sample.cpu().tolist(),
            "entropy_bound": self.gamma,
            "greedy_clean_estimate": self.greedy_clean_estimate,
            "accepted_per_step": accepted_per_step,
        })


@sampler_registry.register("uniform_entropy_bounded")
def _create(config):
    return UniformEntropyBoundedSampler(
        getattr(config.sampling, "entropy_bound", 0.0),
        getattr(config.sampling, "greedy_clean_estimate", False),
    )

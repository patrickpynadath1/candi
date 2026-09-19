from __future__ import annotations

import torch

from .base import Sampler, SamplerSpec, SamplingResult
from .entropy_utils import predictive_entropy, sample_probabilities, select_entropy_bounded
from .registry import sampler_registry
from .utils import continuous_time_grid, preserve_prompt


class CANDIEmbeddingEntropyBoundedSampler(Sampler):
    """Entropy-bounded decoding along a CANDI embedding-space trajectory."""

    spec = SamplerSpec(
        name="candi_embedding_entropy_bounded",
        compatible_families=frozenset({"candi_embedding"}),
        required_capabilities=frozenset({
            "continuous_prior", "continuous_schedule", "continuous_transition",
        }),
    )

    def __init__(self, gamma, greedy_clean_estimate=False):
        if gamma < 0:
            raise ValueError("sampling.entropy_bound must be nonnegative")
        self.gamma = float(gamma)
        self.greedy_clean_estimate = bool(greedy_clean_estimate)

    @torch.no_grad()
    def sample(self, model, request):
        steps, times = continuous_time_grid(model, request)
        rho = 1 - model.noise.alpha_t(times)
        interpolation = model.sampling_continuous_schedule(times)
        state = model.prior_sample(request.num_samples, model.num_tokens)
        tokens = torch.zeros(
            state.shape[:2], dtype=torch.long, device=state.device
        )
        accepted = torch.zeros_like(tokens, dtype=torch.bool)
        if request.prompt_tokens is not None:
            tokens = request.prompt_tokens.clone()
            accepted = request.prompt_mask.bool().clone()
        state = model.sampling_prompt_state(state, tokens, accepted)

        accepted_per_step = []
        nfe_per_sample = torch.zeros(
            state.shape[0], dtype=torch.long, device=state.device
        )
        for index in range(steps):
            active = ~accepted
            if not active.any():
                break
            continued, clean_probs = model.sample_continuous_transition(
                state,
                rho[index],
                rho[index + 1],
                sigma_t=interpolation[index],
                sigma_s=interpolation[index + 1],
                clean_mask=accepted,
                clean_tokens=tokens,
            )
            newly_accepted, counts = select_entropy_bounded(
                predictive_entropy(clean_probs), active, self.gamma
            )
            if self.greedy_clean_estimate:
                clean_tokens = clean_probs.argmax(dim=-1)
            else:
                clean_tokens = sample_probabilities(clean_probs)
            tokens = torch.where(newly_accepted, clean_tokens, tokens)
            accepted |= newly_accepted
            # Accepted sites become exact vocabulary embeddings. All rejected
            # sites retain their current point on the continuous trajectory.
            state = model.sampling_prompt_state(continued, tokens, accepted)
            accepted_per_step.append(counts)
            newly_complete = nfe_per_sample.eq(0) & accepted.all(dim=-1)
            nfe_per_sample[newly_complete] = index + 1

        if not accepted.all():
            raise RuntimeError(
                "CANDI embedding entropy-bounded sampling exhausted "
                "sampling.steps before accepting every position"
            )
        return SamplingResult(
            preserve_prompt(tokens, request),
            {
                "nfe": len(accepted_per_step),
                "nfe_per_sample": nfe_per_sample.cpu().tolist(),
                "entropy_bound": self.gamma,
                "greedy_clean_estimate": self.greedy_clean_estimate,
                "accepted_per_step": accepted_per_step,
            },
        )


@sampler_registry.register("candi_embedding_entropy_bounded")
def _create(config):
    return CANDIEmbeddingEntropyBoundedSampler(
        getattr(config.sampling, "entropy_bound", 0.0),
        getattr(config.sampling, "greedy_clean_estimate", False),
    )

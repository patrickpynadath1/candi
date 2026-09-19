from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Sampler, SamplerSpec, SamplingResult
from .entropy_utils import predictive_entropy, sample_probabilities, select_entropy_bounded
from .registry import sampler_registry
from .utils import continuous_time_grid, inject_onehot_prompt


class CANDIEntropyBoundedSampler(Sampler):
    """Entropy-bounded CANDI with rejected states continuing their trajectory."""

    spec = SamplerSpec(
        name="candi_entropy_bounded",
        compatible_families=frozenset({"candi"}),
        required_capabilities=frozenset({
            "continuous_prior", "continuous_schedule", "candi_discrete_schedule",
            "candi_continuous_transition",
        }),
    )

    def __init__(self, gamma, greedy_clean_estimate=False,
                 entropy_untempered=False):
        if gamma < 0:
            raise ValueError("sampling.entropy_bound must be nonnegative")
        self.gamma = float(gamma)
        self.greedy_clean_estimate = bool(greedy_clean_estimate)
        self.entropy_untempered = bool(entropy_untempered)

    def sample(self, model, request):
        steps, times = continuous_time_grid(model, request)
        x = model.prior_sample(request.num_samples, model.num_tokens)
        prompt_state_fn = getattr(model, "sampling_prompt_state", None)
        if callable(prompt_state_fn):
            x, accepted = prompt_state_fn(
                x, request.prompt_tokens, request.prompt_mask
            )
        else:
            x, accepted = inject_onehot_prompt(x, request)
        accepted = accepted.bool()
        state_tokens_fn = getattr(model, "sampling_state_tokens", None)
        output_tokens = (
            state_tokens_fn(x) if callable(state_tokens_fn) else x.argmax(-1)
        )
        if request.prompt_tokens is not None and request.prompt_mask is not None:
            output_tokens = torch.where(
                request.prompt_mask.bool(), request.prompt_tokens, output_tokens
            )
        sigmas = model.sampling_continuous_schedule(times, steps)
        discrete_times = model.sampling_discrete_schedule(times)
        accepted_per_step = []
        nfe_per_sample = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        for i in range(steps):
            active = ~accepted
            if not active.any():
                break
            conventional = getattr(model, "conventional_sigma_order", False)
            transition = model.sample_candi_continuous_transition(
                x,
                discrete_times[i],
                sigma_s=sigmas[i + 1] if conventional else sigmas[i],
                sigma_t=sigmas[i] if conventional else sigmas[i + 1],
                clean_mask=accepted, time_s=discrete_times[i + 1],
                return_untempered=self.entropy_untempered,
            )
            if self.entropy_untempered:
                continued, clean_probs, entropy_probs = transition
            else:
                continued, clean_probs = transition
                entropy_probs = clean_probs
            entropy = predictive_entropy(entropy_probs)
            newly_accepted, counts = select_entropy_bounded(
                entropy, active, self.gamma
            )
            clean_token_fn = getattr(model, "sampling_clean_tokens", None)
            if self.greedy_clean_estimate:
                clean_tokens = clean_probs.argmax(dim=-1)
            else:
                clean_tokens = (
                    clean_token_fn(clean_probs)
                    if callable(clean_token_fn)
                    else sample_probabilities(clean_probs)
                )
            clean_state_fn = getattr(model, "sampling_clean_state", None)
            clean_state = (
                clean_state_fn(clean_tokens, continued)
                if callable(clean_state_fn)
                else F.one_hot(
                    clean_tokens, num_classes=clean_probs.shape[-1]
                ).to(continued)
            )
            # Crucially, rejected coordinates retain `continued`: their current
            # continuous state after this solver step, rather than fresh noise.
            continued = torch.where(accepted.unsqueeze(-1), x, continued)
            x = torch.where(newly_accepted.unsqueeze(-1), clean_state, continued)
            output_tokens = torch.where(
                newly_accepted, clean_tokens, output_tokens
            )
            accepted |= newly_accepted
            accepted_per_step.append(counts)
            newly_complete = nfe_per_sample.eq(0) & accepted.all(dim=-1)
            nfe_per_sample[newly_complete] = i + 1

        if not accepted.all():
            raise RuntimeError(
                "CANDI entropy-bounded sampling exhausted sampling.steps "
                "before accepting every position"
            )
        nfe = len(accepted_per_step)
        return SamplingResult(output_tokens, {
            "nfe": nfe,
            "nfe_per_sample": nfe_per_sample.cpu().tolist(),
            "entropy_bound": self.gamma,
            "greedy_clean_estimate": self.greedy_clean_estimate,
            "entropy_untempered": self.entropy_untempered,
            "accepted_per_step": accepted_per_step,
        })


@sampler_registry.register("candi_entropy_bounded")
def _create(config):
    return CANDIEntropyBoundedSampler(
        getattr(config.sampling, "entropy_bound", 0.0),
        getattr(config.sampling, "greedy_clean_estimate", False),
        getattr(config.sampling, "entropy_untempered", False),
    )

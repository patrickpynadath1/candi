from __future__ import annotations

import torch

from .base import Sampler, SamplerSpec, SamplingResult
from .registry import sampler_registry
from .entropy_utils import (
    entropy_prefix_size,
    predictive_entropy,
    sample_probabilities,
    select_entropy_bounded,
)
from .utils import apply_token_offset, inject_token_prompt, preserve_prompt


class EntropyBoundedSampler(Sampler):
    """Entropy-bounded parallel unmasking for absorbing-state diffusion.

    At each model evaluation, masked positions are sorted by increasing
    predictive entropy.  The largest prefix U satisfying

        sum(H_i for i in U) - max(H_i for i in U) <= gamma

    is sampled in parallel.  All other positions remain masked and are
    reconsidered after the newly accepted tokens provide more context.
    """

    spec = SamplerSpec(
        name="entropy_bounded",
        compatible_families=frozenset({"absorbing_diffusion"}),
        required_capabilities=frozenset(
            {"discrete_prior", "denoising_log_probs"}
        ),
    )

    def __init__(self, gamma: float, greedy_clean_estimate: bool = False):
        if gamma < 0:
            raise ValueError("sampling.entropy_bound must be nonnegative")
        self.gamma = float(gamma)
        self.greedy_clean_estimate = bool(greedy_clean_estimate)

    @staticmethod
    def _prefix_size(entropy: torch.Tensor, gamma: float) -> int:
        return entropy_prefix_size(entropy, gamma)

    def sample(self, model, request):
        x = inject_token_prompt(
            model.prior_sample(request.num_samples, model.num_tokens), request
        )
        prompt_mask = (
            torch.zeros_like(x, dtype=torch.bool)
            if request.prompt_mask is None
            else request.prompt_mask.bool()
        )
        generated_mask = ~prompt_mask
        nfe = 0
        accepted_per_step: list[list[int]] = []
        nfe_per_sample = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        max_nfe = int(request.options.get("max_nfe", model.num_tokens))

        while True:
            masked = (x == model.mask_index) & generated_mask
            remaining = masked.sum(dim=-1)
            if not remaining.any():
                break
            if nfe >= max_nfe:
                raise RuntimeError(
                    "entropy-bounded sampling exceeded max_nfe before all "
                    "positions were unmasked; increase request.options['max_nfe']"
                )

            # This is meaningful for time-conditioned absorbing models and is
            # ignored by time-independent MDLMs.
            denominator = generated_mask.sum(dim=-1).clamp_min(1)
            t = (remaining / denominator).to(model.dtype).unsqueeze(-1)
            log_probs = model.sample_denoising_log_probs(x=x, t=t)
            # Entropy and categorical sampling are more stable in fp32 than
            # in the bf16 precision commonly used by the denoiser.
            probs = log_probs.float().exp()
            entropy = predictive_entropy(probs)

            sampled = (
                probs.argmax(dim=-1)
                if self.greedy_clean_estimate
                else sample_probabilities(probs)
            )
            accepted, counts = select_entropy_bounded(
                entropy, masked, self.gamma
            )

            x = torch.where(accepted, sampled, x)
            x = preserve_prompt(x, request)
            accepted_per_step.append(counts)
            nfe += 1
            newly_complete = (
                nfe_per_sample.eq(0)
                & ~((x == model.mask_index) & generated_mask).any(dim=-1)
            )
            nfe_per_sample[newly_complete] = nfe

        return SamplingResult(
            apply_token_offset(model, x),
            {
                "nfe": nfe,
                "nfe_per_sample": nfe_per_sample.cpu().tolist(),
                "entropy_bound": self.gamma,
                "greedy_clean_estimate": self.greedy_clean_estimate,
                "accepted_per_step": accepted_per_step,
            },
        )


@sampler_registry.register("entropy_bounded")
def _create(config):
    return EntropyBoundedSampler(
        getattr(config.sampling, "entropy_bound", 0.0),
        getattr(config.sampling, "greedy_clean_estimate", False),
    )

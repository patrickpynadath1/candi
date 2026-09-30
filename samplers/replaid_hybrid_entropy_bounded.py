"""Entropy-bounded reveals on the hybrid Gaussian clock (inference heuristic)."""

import torch

from .base import Sampler, SamplerSpec, SamplingResult
from .entropy_utils import predictive_entropy, sample_probabilities, select_entropy_bounded
from .registry import sampler_registry
from .replaid_ddpm import RePlaidDDPMSampler


class RePlaidHybridEntropyBoundedSampler(Sampler):
    """Continue latent states while accepting a bounded low-entropy prefix."""

    spec = SamplerSpec(
        name="replaid_hybrid_entropy_bounded",
        compatible_families=frozenset({"replaid_hybrid"}),
        required_capabilities=frozenset(
            {
                "replaid_prior",
                "hybrid_denoise",
                "hybrid_clock",
                "embedding_projection",
            }
        ),
    )

    def __init__(self, gamma):
        if gamma < 0:
            raise ValueError("sampling.entropy_bound must be nonnegative")
        self.gamma = float(gamma)

    @torch.no_grad()
    def sample(self, model, request):
        steps = request.num_steps or model.config.sampling.steps
        if steps < 1 or request.eps != 0:
            raise ValueError("hybrid EB requires positive steps and eps=0")
        times = torch.linspace(1, request.eps, steps + 1, device=model.device)
        z_t = model.prior_sample(request.num_samples, model.num_tokens)
        kept = torch.zeros(z_t.shape[:2], device=model.device, dtype=torch.bool)
        tokens = torch.zeros(z_t.shape[:2], device=model.device, dtype=torch.long)
        if request.prompt_mask is not None:
            kept |= request.prompt_mask.bool()
            tokens = torch.where(kept, request.prompt_tokens, tokens)
            z_t = torch.where(
                kept.unsqueeze(-1), model.sampling_embeddings(tokens), z_t
            )

        accepted_per_step = []
        nfe_per_sample = torch.zeros(
            request.num_samples, dtype=torch.long, device=model.device
        )
        for index in range(steps):
            active = ~kept
            if not active.any():
                break
            t = times[index].expand(request.num_samples)
            s = times[index + 1].expand(request.num_samples)
            log_probs, predicted, gamma_t = model.sample_hybrid_denoise(
                z_t, t, tokens, kept
            )
            probs = log_probs.float().exp()
            entropy = predictive_entropy(probs)
            newly_kept, counts = select_entropy_bounded(
                entropy, active, self.gamma
            )
            proposed = sample_probabilities(probs)
            tokens = torch.where(newly_kept, proposed, tokens)

            gamma_s = model.sampling_gamma(s)
            noise = torch.randn_like(z_t)
            continued = RePlaidDDPMSampler.transition(
                z_t, predicted, gamma_t, gamma_s, noise
            )
            # Previously and newly accepted coordinates are exact points from
            # the shared RePlaid embedding table. Rejected coordinates retain
            # their continued Gaussian state.
            kept |= newly_kept
            hard = model.sampling_embeddings(tokens)
            z_t = torch.where(kept.unsqueeze(-1), hard, continued)
            accepted_per_step.append(counts)
            newly_complete = nfe_per_sample.eq(0) & kept.all(dim=-1)
            nfe_per_sample[newly_complete] = index + 1

        if not kept.all():
            raise RuntimeError(
                "RePlaid entropy-bounded sampling exhausted sampling.steps "
                "before accepting every position"
            )
        nfe = len(accepted_per_step)
        return SamplingResult(
            tokens,
            {
                "nfe": nfe,
                "schedule": "fixed_learned_clock",
                "nfe_per_sample": nfe_per_sample.cpu().tolist(),
                "entropy_bound": self.gamma,
                "accepted_per_step": accepted_per_step,
            },
        )


@sampler_registry.register("replaid_hybrid_entropy_bounded")
def _create(config):
    return RePlaidHybridEntropyBoundedSampler(
        getattr(config.sampling, "entropy_bound", 0.0)
    )

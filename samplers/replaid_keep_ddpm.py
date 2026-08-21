"""Hard-reveal sampler for keep-conditioned RePlaid variants."""

import torch

from .base import Sampler, SamplerSpec, SamplingResult
from .registry import sampler_registry
from .replaid_ddpm import RePlaidDDPMSampler


class RePlaidKeepDDPMSampler(Sampler):
    spec = SamplerSpec(
        name="replaid_keep_ddpm",
        compatible_families=frozenset({"replaid_keep"}),
        required_capabilities=frozenset(
            {
                "replaid_prior",
                "replaid_keep_denoise",
                "replaid_schedule",
                "embedding_projection",
            }
        ),
    )

    @staticmethod
    def reveal_probability(gamma_t, gamma_s):
        keep_t = torch.sigmoid(-gamma_t)
        keep_s = torch.sigmoid(-gamma_s)
        return ((keep_s - keep_t) / (1 - keep_t).clamp_min(1e-12)).clamp(0, 1)

    @staticmethod
    def choose_tokens(log_probs, strategy):
        if strategy == "argmax":
            return log_probs.argmax(-1)
        if strategy != "sample":
            raise ValueError("reveal_strategy must be 'sample' or 'argmax'")
        flat = log_probs.exp().reshape(-1, log_probs.shape[-1])
        return torch.multinomial(flat, 1).reshape(log_probs.shape[:-1])

    @torch.no_grad()
    def sample(self, model, request):
        steps = request.num_steps or model.config.sampling.steps
        times = torch.linspace(1, request.eps, steps + 1, device=model.device)
        z_t = model.prior_sample(request.num_samples, model.num_tokens)
        kept = torch.zeros(z_t.shape[:2], device=model.device, dtype=torch.bool)
        tokens = torch.zeros(z_t.shape[:2], device=model.device, dtype=torch.long)
        if request.prompt_mask is not None:
            kept |= request.prompt_mask.bool()
            tokens = torch.where(kept, request.prompt_tokens, tokens)
            prompt_embeddings = model.sampling_embeddings(tokens)
            z_t = torch.where(kept.unsqueeze(-1), prompt_embeddings, z_t)

        strategy = model.config.algo.reveal_strategy
        for index in range(steps):
            t = times[index].expand(request.num_samples)
            s = times[index + 1].expand(request.num_samples)
            log_probs, predicted, gamma_t = model.sample_replaid_keep_denoise(
                z_t, t, tokens, kept
            )
            gamma_s = model.sampling_gamma(s)
            proposed = self.choose_tokens(log_probs, strategy)
            probability = self.reveal_probability(gamma_t, gamma_s)[:, None]
            reveal = (~kept) & (torch.rand_like(probability.expand_as(kept)) < probability)
            if index == steps - 1:
                reveal |= ~kept
            tokens = torch.where(reveal, proposed, tokens)
            noise = torch.zeros_like(z_t) if index == steps - 1 else torch.randn_like(z_t)
            z_t = RePlaidDDPMSampler.transition(
                z_t, predicted, gamma_t, gamma_s, noise
            )
            kept |= reveal
            hard_embeddings = model.sampling_embeddings(tokens)
            z_t = torch.where(kept.unsqueeze(-1), hard_embeddings, z_t)
        return SamplingResult(tokens, {"nfe": steps})


@sampler_registry.register("replaid_keep_ddpm")
def _create(_config):
    return RePlaidKeepDDPMSampler()


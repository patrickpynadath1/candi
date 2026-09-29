"""Ancestral reverse hybrid process with the learned shared clock."""
import torch

from .base import Sampler, SamplerSpec, SamplingResult
from .registry import sampler_registry
from .replaid_ddpm import RePlaidDDPMSampler
from .replaid_keep_ddpm import RePlaidKeepDDPMSampler


class RePlaidHybridDDPMSampler(Sampler):
    spec = SamplerSpec(name="replaid_hybrid_ddpm", compatible_families=frozenset({"replaid_hybrid"}),
                       required_capabilities=frozenset({"replaid_prior", "hybrid_denoise", "hybrid_clock", "embedding_projection"}))

    @staticmethod
    def reveal_probability(u_t, u_s):
        return ((u_t-u_s)/u_t.clamp_min(torch.finfo(u_t.dtype).tiny)).clamp(0, 1)

    @torch.no_grad()
    def sample(self, model, request):
        steps = request.num_steps or model.config.sampling.steps
        if steps < 1 or request.eps != 0:
            raise ValueError("hybrid sampling requires positive steps and eps=0 (all tokens revealed)")
        times = torch.linspace(1, 0, steps+1, device=model.device, dtype=torch.float64)
        z = model.prior_sample(request.num_samples, model.num_tokens)
        kept = torch.zeros(z.shape[:2], device=z.device, dtype=torch.bool)
        tokens = torch.zeros(z.shape[:2], device=z.device, dtype=torch.long)
        if request.prompt_mask is not None:
            kept |= request.prompt_mask.bool()
            tokens = torch.where(kept, request.prompt_tokens, tokens)
            z = torch.where(kept[..., None], model.sampling_embeddings(tokens), z)
        for index in range(steps):
            t, s = times[index].expand(z.shape[0]), times[index+1].expand(z.shape[0])
            log_probs, predicted, gamma_t = model.sample_hybrid_denoise(z, t, tokens, kept)
            u_t = model.sampling_mask_probability(t)
            u_s = model.sampling_mask_probability(s)
            probability = self.reveal_probability(u_t, u_s)
            reveal = ~kept & (torch.rand(kept.shape, device=z.device) < probability[:, None])
            proposed = RePlaidKeepDDPMSampler.choose_tokens(log_probs, "sample")
            tokens = torch.where(reveal, proposed, tokens)
            z = RePlaidDDPMSampler.transition(z, predicted, gamma_t, model.sampling_gamma(s), torch.randn_like(z))
            kept |= reveal
            z = torch.where(kept[..., None], model.sampling_embeddings(tokens), z)
        return SamplingResult(tokens, {"nfe": steps})


@sampler_registry.register("replaid_hybrid_ddpm")
def _create(_config):
    return RePlaidHybridDDPMSampler()

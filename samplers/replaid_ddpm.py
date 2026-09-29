"""Stochastic ancestral sampler from RePlaid Appendix D.1."""

import torch

from .base import Sampler, SamplerSpec, SamplingResult
from .registry import sampler_registry
from .utils import preserve_prompt


class RePlaidDDPMSampler(Sampler):
    spec = SamplerSpec(
        name="replaid_ddpm",
        compatible_families=frozenset({"replaid"}),
        required_capabilities=frozenset(
            {
                "replaid_prior",
                "replaid_denoise",
                "replaid_schedule",
                "embedding_projection",
            }
        ),
    )

    @staticmethod
    def transition(z_t, clean, gamma_t, gamma_s, noise):
        alpha_t = torch.sigmoid(-gamma_t).sqrt().to(z_t)
        alpha_s = torch.sigmoid(-gamma_s).sqrt().to(z_t)
        c = 1 - torch.exp(gamma_s - gamma_t).to(z_t)
        while alpha_t.ndim < z_t.ndim:
            alpha_t = alpha_t.unsqueeze(-1)
            alpha_s = alpha_s.unsqueeze(-1)
            c = c.unsqueeze(-1)
        return (
            (alpha_s / alpha_t) * (1 - c) * z_t
            + c * alpha_s * clean
            + torch.sqrt(c * (1 - alpha_s.square()).clamp_min(0)) * noise
        )

    @torch.no_grad()
    def sample(self, model, request):
        steps = request.num_steps or model.config.sampling.steps
        times = torch.linspace(1, request.eps, steps + 1, device=model.device)
        z_t = model.prior_sample(request.num_samples, model.num_tokens)
        prompt_mask = request.prompt_mask
        prompt_embeddings = prompt_noise = None
        if prompt_mask is not None:
            prompt_mask = prompt_mask.bool().unsqueeze(-1)
            prompt_embeddings = model.sampling_embeddings(request.prompt_tokens)
            prompt_noise = torch.randn_like(prompt_embeddings)

            def clamp_prompt(state, gamma):
                if model.config.algo.get("clean_context", False):
                    return torch.where(prompt_mask, prompt_embeddings, state)
                alpha = torch.sigmoid(-gamma).sqrt().to(state)[:, None, None]
                sigma = torch.sigmoid(gamma).sqrt().to(state)[:, None, None]
                forward_prompt = alpha * prompt_embeddings + sigma * prompt_noise
                return torch.where(prompt_mask, forward_prompt, state)

            initial_gamma = model.sampling_gamma(
                times[0].expand(request.num_samples)
            )
            z_t = clamp_prompt(z_t, initial_gamma)
        else:
            clamp_prompt = None
        self_condition = None
        nfe = 0
        use_self_condition = model.config.algo.get("self_condition_rate", 0.25) > 0
        sc_mode = model.config.algo.get("sampling_self_condition", "previous")
        if sc_mode not in {"bootstrap", "previous"}:
            raise ValueError(f"unknown sampling self-conditioning mode: {sc_mode}")

        def denoise_at(state, time):
            nonlocal nfe
            context = self_condition
            if use_self_condition and sc_mode == "bootstrap":
                _, context, _ = model.sample_replaid_denoise(state, time, None)
                nfe += 1
                if prompt_mask is not None:
                    context = torch.where(prompt_mask, prompt_embeddings, context)
            result = model.sample_replaid_denoise(state, time, context)
            nfe += 1
            return result

        for index in range(steps):
            t = times[index].expand(request.num_samples)
            s = times[index + 1].expand(request.num_samples)
            log_probs, clean, gamma_t = denoise_at(z_t, t)
            gamma_s = model.sampling_gamma(s)
            # The last transition still ends at nonzero sigma_0; Appendix D.1
            # includes posterior noise here too.
            noise = torch.randn_like(z_t)
            z_t = self.transition(z_t, clean, gamma_t, gamma_s, noise)
            self_condition = clean if use_self_condition else None
            if clamp_prompt is not None:
                z_t = clamp_prompt(z_t, gamma_s)
                if self_condition is not None:
                    self_condition = torch.where(
                        prompt_mask, prompt_embeddings, self_condition
                    )
        final, _, _ = denoise_at(z_t, times[-1].expand(request.num_samples))
        tokens = preserve_prompt(final.argmax(-1), request)
        return SamplingResult(tokens, {"nfe": nfe})


@sampler_registry.register("replaid_ddpm")
def _create(_config):
    return RePlaidDDPMSampler()

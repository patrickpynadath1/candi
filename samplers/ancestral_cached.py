import torch

from .base import Sampler, SamplerSpec, SamplingResult
from .registry import sampler_registry
from .utils import (
    apply_token_offset,
    diffusion_time_grid,
    inject_token_prompt,
    preserve_prompt,
)


class CachedAncestralSampler(Sampler):
    def __init__(self, noise_removal, greedy_clean_estimate=False):
        self.noise_removal = noise_removal
        self.greedy_clean_estimate = bool(greedy_clean_estimate)
        required = {"discrete_prior", "noise_schedule", "ancestral_transition"}
        if noise_removal == "greedy":
            required.add("greedy_denoise")
        self.spec = SamplerSpec(
            name="ancestral_cache",
            compatible_families=frozenset(
                {"absorbing_diffusion", "uniform_diffusion"}
            ),
            required_capabilities=frozenset(required),
        )

    def sample(self, model, request):
        steps, times, dt = diffusion_time_grid(model, request)
        x = inject_token_prompt(
            model.prior_sample(request.num_samples, model.num_tokens), request
        )
        cache = None
        for i in range(steps):
            t = times[i] * torch.ones(x.shape[0], 1, device=model.device)
            transition_args = dict(
                x=x, t=t, dt=dt, p_x0=cache, prompt_mask=request.prompt_mask
            )
            if model.sampling_family == "uniform_diffusion":
                transition_args["greedy_clean_estimate"] = self.greedy_clean_estimate
            cache, x_next = model.sample_ancestral_transition(**transition_args)
            if (not torch.allclose(x_next, x) or model.time_conditioning
                    or hasattr(getattr(model, "backbone", None), "latent_self_conditioning")):
                cache = None
            x = x_next
        nfe = steps
        t0 = times[-1] * torch.ones(x.shape[0], 1, device=model.device)
        if self.noise_removal == "ancestral":
            _, x = model.sample_ancestral_transition(
                x=x, t=t0, dt=None, p_x0=cache, noise_removal_step=True,
                prompt_mask=request.prompt_mask,
            )
            nfe += 1
        elif self.noise_removal == "greedy":
            x = model.sample_greedy_denoise(x=x, t=t0)
            x = preserve_prompt(x, request)
            nfe += 1
        return SamplingResult(apply_token_offset(model, x), {"nfe": nfe})


@sampler_registry.register("ancestral_cache")
def _create(config):
    return CachedAncestralSampler(
        config.sampling.noise_removal,
        getattr(config.sampling, "greedy_clean_estimate", False),
    )

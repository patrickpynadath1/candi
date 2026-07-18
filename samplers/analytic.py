import torch

from .base import Sampler, SamplerSpec, SamplingResult
from .registry import sampler_registry
from .utils import apply_token_offset, diffusion_time_grid, inject_token_prompt


class AnalyticSampler(Sampler):
    def __init__(self, noise_removal):
        self.noise_removal = noise_removal
        required = {"discrete_prior", "noise_schedule", "analytic_transition"}
        if noise_removal == "ancestral":
            required.add("denoiser_transition")
        elif noise_removal == "greedy":
            required.add("greedy_denoise")
        self.spec = SamplerSpec(
            name="analytic",
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
        for i in range(steps):
            t = times[i] * torch.ones(x.shape[0], 1, device=model.device)
            x = model.sample_analytic_transition(x=x, t=t, dt=dt)
        nfe = steps
        t0 = times[-1] * torch.ones(x.shape[0], 1, device=model.device)
        if self.noise_removal == "ancestral":
            x = model.sample_denoiser_transition(x=x, t=t0)
            nfe += 1
        elif self.noise_removal == "greedy":
            x = model.sample_greedy_denoise(x=x, t=t0)
            nfe += 1
        return SamplingResult(apply_token_offset(model, x), {"nfe": nfe})


@sampler_registry.register("analytic")
def _create(config):
    return AnalyticSampler(config.sampling.noise_removal)

"""Paper I.1/I.2 baselines with exact linear/cosine/adaptive time grids."""

import torch
from .base import Sampler, SamplerSpec, SamplingResult
from .registry import sampler_registry
from .utils import preserve_prompt


class PaperDiscreteSampler(Sampler):
    spec = SamplerSpec(
        "paper_discrete",
        frozenset({"absorbing_diffusion", "uniform_diffusion"}),
        frozenset(
            {
                "discrete_prior",
                "denoising_log_probs",
                "discrete_reverse_probs",
                "discrete_corruption",
            }
        ),
    )

    def __init__(self, config):
        self.schedule = config.sampling.get("time_grid", "linear")
        self.pc = config.sampling.get("predict_correct", False)

    def sample(self, model, request):
        steps = request.num_steps or model.config.sampling.steps
        times = model.paper_time_sampler.sampling_grid(
            steps, self.schedule, model.device
        )
        x = preserve_prompt(
            model.prior_sample(request.num_samples, model.num_tokens), request
        )
        for i in range(steps):
            t = times[i].expand(request.num_samples, 1)
            s = times[i + 1].expand_as(t)
            probs = model.sample_denoising_log_probs(x, t).exp()
            if self.pc or i == steps - 1:
                clean = torch.distributions.Categorical(probs=probs).sample()
                x = clean if i == steps - 1 else model.q_xt(clean, 1 - s)
            else:
                posterior = model.reverse_transition_probs(probs, x, 1 - s, 1 - t)
                x = torch.distributions.Categorical(probs=posterior).sample()
            x = preserve_prompt(x, request)
        return SamplingResult(x, {"nfe": steps})


@sampler_registry.register("paper_discrete")
def _create(config):
    return PaperDiscreteSampler(config)

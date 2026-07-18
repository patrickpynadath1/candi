import torch

from .base import Sampler, SamplerSpec, SamplingResult
from .registry import sampler_registry
from .utils import preserve_prompt


class SemicatEulerSampler(Sampler):
    spec = SamplerSpec(
        name="semicat_euler",
        compatible_families=frozenset({"semicat"}),
        required_capabilities=frozenset(
            {"continuous_prior", "semicat_vector_field"}
        ),
    )

    def sample(self, model, request):
        steps = request.num_steps or model.config.sampling.steps
        x = model.prior_sample(request.num_samples, model.num_tokens)
        times = torch.linspace(0, 1, steps + 1, device=model.device)
        for start, end in zip(times[:-1], times[1:]):
            velocity = model.sample_semicat_vector_field(
                start.expand(request.num_samples), x
            )
            x = x + (end - start) * velocity
        return SamplingResult(
            preserve_prompt(x.argmax(-1), request),
            {"nfe": steps},
        )


@sampler_registry.register("semicat_euler")
def _create(_config):
    return SemicatEulerSampler()

import torch

from .base import Sampler, SamplerSpec, SamplingResult
from .registry import sampler_registry
from .utils import preserve_prompt


class SemicatFlowMapSampler(Sampler):
    spec = SamplerSpec(
        name="semicat_flow_map",
        compatible_families=frozenset({"semicat"}),
        required_capabilities=frozenset({"continuous_prior", "semicat_flow_map"}),
    )

    def sample(self, model, request):
        steps = request.num_steps or model.config.sampling.steps
        x = model.prior_sample(request.num_samples, model.num_tokens)
        times = torch.linspace(0, 1, steps + 1, device=model.device)
        for start, end in zip(times[:-1], times[1:]):
            x = model.sample_semicat_flow_map(
                x, start.expand(request.num_samples), end.expand(request.num_samples)
            )
        return SamplingResult(
            preserve_prompt(x.argmax(-1), request),
            {"nfe": steps},
        )


@sampler_registry.register("semicat_flow_map")
def _create(_config):
    return SemicatFlowMapSampler()

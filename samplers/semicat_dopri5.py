import torch
from torchdiffeq import odeint

from .base import Sampler, SamplerSpec, SamplingResult
from .registry import sampler_registry
from .utils import preserve_prompt


class SemicatDopri5Sampler(Sampler):
    spec = SamplerSpec(
        name="semicat_dopri5",
        compatible_families=frozenset({"semicat"}),
        required_capabilities=frozenset(
            {"continuous_prior", "semicat_vector_field"}
        ),
    )

    def sample(self, model, request):
        x = model.prior_sample(request.num_samples, model.num_tokens)
        x = odeint(
            func=model.sample_semicat_vector_field,
            y0=x,
            t=torch.tensor([0.0, 1.0], device=model.device),
            method="dopri5",
            **request.options,
        )[-1]
        return SamplingResult(
            preserve_prompt(x.argmax(-1), request),
            {"nfe": None},
        )


@sampler_registry.register("semicat_dopri5")
def _create(_config):
    return SemicatDopri5Sampler()

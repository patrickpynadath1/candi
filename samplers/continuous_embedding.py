from .base import Sampler, SamplerSpec, SamplingResult
from .registry import sampler_registry
from .utils import run_continuous_diffusion


class ContinuousEmbeddingSampler(Sampler):
    spec = SamplerSpec(
        name="continuous_embedding",
        compatible_families=frozenset({"continuous_embedding"}),
        required_capabilities=frozenset(
            {"continuous_prior", "continuous_schedule", "continuous_transition"}
        ),
    )

    def sample(self, model, request):
        samples, nfe = run_continuous_diffusion(model, request)
        return SamplingResult(samples, {"nfe": nfe})


@sampler_registry.register("continuous_embedding")
def _create(_config):
    return ContinuousEmbeddingSampler()

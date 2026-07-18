from .base import Sampler, SamplerSpec, SamplingResult
from .registry import sampler_registry
from .utils import continuous_time_grid, inject_onehot_prompt


class CachedCANDISampler(Sampler):
    spec = SamplerSpec(
        name="candi_cached",
        compatible_families=frozenset({"candi"}),
        required_capabilities=frozenset({
            "continuous_prior", "continuous_schedule", "embedding_projection",
            "candi_cached_continuous_transition",
            "candi_cached_discrete_transition",
        }),
    )

    def sample(self, model, request):
        steps, times = continuous_time_grid(model, request)
        x, clean = inject_onehot_prompt(
            model.prior_sample(request.num_samples, model.num_tokens), request
        )
        sigmas = model.sampling_continuous_schedule(times, steps)
        embedding_cache = model.sampling_embeddings(x)
        x = x.argmax(-1)
        dt = (1 - request.eps) / steps
        for i in range(steps):
            embedding_cache, predicted = model.sample_candi_cached_continuous_transition(
                x=x, time_t=times[i], time_s=times[i + 1],
                sigma_s=sigmas[i], sigma_t=sigmas[i + 1],
                clean_mask=clean.float(), embedding_cache=embedding_cache,
                is_first=i == 0,
            )
            x, clean = model.sample_candi_cached_discrete_transition(
                predicted, x, times[i], dt, prev_clean_mask=clean,
                noise_removal_step=False,
            )
        return SamplingResult(x, {"nfe": steps})


@sampler_registry.register("candi_cached")
def _create(_config):
    return CachedCANDISampler()

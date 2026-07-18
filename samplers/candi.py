from .base import Sampler, SamplerSpec, SamplingResult
from .registry import sampler_registry
from .utils import continuous_time_grid, inject_onehot_prompt


class CANDISampler(Sampler):
    spec = SamplerSpec(
        name="candi",
        compatible_families=frozenset({"candi"}),
        required_capabilities=frozenset({
            "continuous_prior", "continuous_schedule",
            "candi_continuous_transition", "candi_discrete_transition",
        }),
    )

    def sample(self, model, request):
        steps, times = continuous_time_grid(model, request)
        x, clean = inject_onehot_prompt(
            model.prior_sample(request.num_samples, model.num_tokens), request
        )
        sigmas = model.sampling_continuous_schedule(times, steps)
        dt = (1 - request.eps) / steps
        for i in range(steps):
            continuous, clean_probs = model.sample_candi_continuous_transition(
                x, times[i], sigma_s=sigmas[i], sigma_t=sigmas[i + 1],
                clean_mask=clean, time_s=times[i + 1],
            )
            x, clean = model.sample_candi_discrete_transition(
                continuous, clean_probs, times[i], dt, prev_clean_mask=clean
            )
        return SamplingResult(x.argmax(-1), {"nfe": steps})


@sampler_registry.register("candi")
def _create(_config):
    return CANDISampler()

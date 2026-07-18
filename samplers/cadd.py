import torch

from .base import Sampler, SamplerSpec, SamplingResult
from .registry import sampler_registry
from .utils import continuous_time_grid, inject_onehot_prompt


class CADDSampler(Sampler):
    spec = SamplerSpec(
        name="cadd",
        compatible_families=frozenset({"cadd"}),
        required_capabilities=frozenset(
            {"continuous_prior", "cadd_continuous_transition", "cadd_discrete_transition"}
        ),
    )

    def sample(self, model, request):
        steps, times = continuous_time_grid(model, request)
        embeds, clean = inject_onehot_prompt(
            model.prior_sample(request.num_samples, model.num_tokens), request
        )
        tokens = torch.full(
            embeds.shape[:2], model.mask_index, device=embeds.device, dtype=torch.long
        )
        if request.prompt_tokens is not None and request.prompt_mask is not None:
            tokens = torch.where(request.prompt_mask.bool(), request.prompt_tokens, tokens)
        dt = (1 - request.eps) / steps
        for i in range(steps):
            embeds, predicted = model.sample_cadd_continuous_transition(
                xt_embeds=embeds, xt_tokens=tokens, time_t=times[i],
                time_s=times[i + 1], clean_mask=clean,
            )
            embeds, tokens, clean = model.sample_cadd_discrete_transition(
                x0_hat_tokens=predicted, xt_tokens=tokens, xt_embeds=embeds,
                t=times[i], dt=dt, prev_clean_mask=clean,
                noise_removal_step=i == steps - 1,
            )
        return SamplingResult(tokens, {"nfe": steps})


@sampler_registry.register("cadd")
def _create(_config):
    return CADDSampler()

import torch

from .base import Sampler, SamplerSpec, SamplingResult
from .registry import sampler_registry
from .utils import preserve_prompt


class LatentFlowMatchingSampler(Sampler):
    spec = SamplerSpec(
        name="latent_fm",
        compatible_families=frozenset({"latent_fm"}),
        required_capabilities=frozenset({"latent_prior", "latent_flow_map"}),
    )

    def sample(self, model, request):
        steps = request.num_steps or model.config.sampling.steps
        latent = model.prior_sample(request.num_samples, model.num_tokens)
        times = torch.linspace(0, 1, steps + 1, device=model.device)
        mode = request.options.get(
            "feedback_mode", model.config.sampling.get("latent_feedback", None)
        )
        temperature = request.options.get(
            "feedback_temperature",
            model.config.sampling.get("feedback_temperature", None),
        )
        logits = None
        for start, end in zip(times[:-1], times[1:]):
            latent, logits = model.sample_latent_flow_map(
                latent,
                start.expand(request.num_samples),
                end.expand(request.num_samples),
                prompt_tokens=request.prompt_tokens,
                prompt_mask=request.prompt_mask,
                feedback_mode=mode,
                feedback_temperature=temperature,
            )
        samples = logits.argmax(-1)
        return SamplingResult(
            preserve_prompt(samples, request),
            {"nfe": 2 * steps, "decoder_nfe": steps, "encoder_nfe": steps},
        )


@sampler_registry.register("latent_fm")
def _create(_config):
    return LatentFlowMatchingSampler()


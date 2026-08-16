from __future__ import annotations

import torch

from .base import Sampler, SamplerSpec, SamplingRequest, SamplingResult
from .registry import sampler_registry
from .utils import autoregressive_prompt_length


class AutoregressiveSampler(Sampler):
    spec = SamplerSpec(
        name="ar",
        compatible_families=frozenset({"autoregressive"}),
        required_capabilities=frozenset({"next_token_logits"}),
    )

    def __init__(self, *, use_float64: bool, temperature: float):
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        self.use_float64 = use_float64
        self.temperature = temperature

    def sample(self, model, request: SamplingRequest) -> SamplingResult:
        num_pred_tokens = model.num_tokens - 1
        x = torch.zeros(
            (request.num_samples, model.num_tokens),
            dtype=torch.long,
            device=model.device,
        )

        prompt_length = autoregressive_prompt_length(request)
        if prompt_length is not None:
            x = torch.where(request.prompt_mask.bool(), request.prompt_tokens, x)
            start = max(prompt_length - 1, 0)
        else:
            start = 0
            x[:, 0] = model.tokenizer.bos_token_id

        nfe = 0
        finished = torch.zeros(
            request.num_samples, dtype=torch.bool, device=model.device
        )
        eos = getattr(model.tokenizer, "eos_token_id", None)
        for i in range(start, num_pred_tokens):
            logits = model.next_token_logits(x[:, : i + 1])
            step_logits = logits[:, -1]
            if self.temperature == 0:
                next_tokens = step_logits.argmax(-1)
            else:
                step_log_probs = (step_logits / self.temperature).log_softmax(-1)
                noise = -torch.empty_like(step_log_probs).exponential_().log()
                if self.use_float64:
                    noise = noise.to(torch.float64)
                next_tokens = (step_log_probs + noise).argmax(-1)
            x[:, i + 1] = torch.where(finished, x[:, i + 1], next_tokens)
            if eos is not None:
                finished |= next_tokens == eos
            nfe += 1
            if eos is not None and finished.all():
                break
        return SamplingResult(x, {"nfe": nfe})


@sampler_registry.register("ar")
def _create_ar(config):
    algo_config = getattr(config, "algo", None)
    temperature = algo_config.get("temp", 1.0) if algo_config is not None else 1.0
    return AutoregressiveSampler(
        use_float64=config.sampling.use_float64,
        temperature=temperature,
    )

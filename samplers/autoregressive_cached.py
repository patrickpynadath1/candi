from __future__ import annotations

import torch

from .base import Sampler, SamplerSpec, SamplingRequest, SamplingResult
from .registry import sampler_registry
from .utils import autoregressive_prompt_length


class CachedAutoregressiveSampler(Sampler):
    spec = SamplerSpec(
        name="ar_cached",
        compatible_families=frozenset({"autoregressive"}),
        required_capabilities=frozenset({"cached_next_token_logits"}),
    )

    def __init__(self, *, use_float64: bool, temperature: float):
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        self.use_float64 = use_float64
        self.temperature = temperature

    @torch.no_grad()
    def sample(self, model, request: SamplingRequest) -> SamplingResult:
        num_pred_tokens = model.num_tokens - 1
        samples = torch.zeros(
            (request.num_samples, model.num_tokens),
            dtype=torch.long,
            device=model.device,
        )
        prompt_length = autoregressive_prompt_length(request)
        if prompt_length is not None:
            samples = torch.where(
                request.prompt_mask.bool(), request.prompt_tokens, samples
            )
            start = max(prompt_length - 1, 0)
        else:
            start = 0
            samples[:, 0] = model.tokenizer.bos_token_id

        cache = None
        model_calls = 0
        tokens_processed = 0
        finished = torch.zeros(
            request.num_samples, dtype=torch.bool, device=model.device
        )
        eos = getattr(model.tokenizer, "eos_token_id", None)
        for index in range(start, num_pred_tokens):
            if cache is None:
                tokens = samples[:, : index + 1]
            else:
                tokens = samples[:, index : index + 1]
            logits, cache = model.next_token_logits_cached(tokens, cache=cache)
            step_logits = logits[:, -1]
            if self.temperature == 0:
                next_tokens = step_logits.argmax(-1)
            else:
                log_probs = (step_logits / self.temperature).log_softmax(-1)
                noise = -torch.empty_like(log_probs).exponential_().log()
                if self.use_float64:
                    noise = noise.to(torch.float64)
                next_tokens = (log_probs + noise).argmax(-1)
            samples[:, index + 1] = torch.where(
                finished, samples[:, index + 1], next_tokens
            )
            if eos is not None:
                finished |= next_tokens == eos
            model_calls += 1
            tokens_processed += tokens.shape[1]
            if eos is not None and finished.all():
                break
        return SamplingResult(
            samples,
            {
                "nfe": model_calls,
                "tokens_processed": tokens_processed,
                "cache_used": True,
            },
        )


@sampler_registry.register("ar_cached")
def _create_cached_ar(config):
    algo_config = getattr(config, "algo", None)
    temperature = algo_config.get("temp", 1.0) if algo_config is not None else 1.0
    return CachedAutoregressiveSampler(
        use_float64=config.sampling.use_float64,
        temperature=temperature,
    )

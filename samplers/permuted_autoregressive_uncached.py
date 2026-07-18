from __future__ import annotations

import torch

from .base import Sampler, SamplerSpec, SamplingRequest, SamplingResult
from .permuted_autoregressive import generation_order
from .registry import sampler_registry


class UncachedPermutedAutoregressiveSampler(Sampler):
    spec = SamplerSpec(
        name="permuted_ar_uncached",
        compatible_families=frozenset({"permuted_autoregressive"}),
        required_capabilities=frozenset({"permuted_logits"}),
    )

    def __init__(self, *, use_float64: bool, temperature: float):
        self.use_float64 = use_float64
        self.temperature = temperature

    @torch.no_grad()
    def sample(self, model, request: SamplingRequest) -> SamplingResult:
        if request.prompt_tokens is not None or request.prompt_mask is not None:
            raise ValueError("prompted any-order sampling is not yet supported")
        order = generation_order(model, request)
        mask_id = getattr(model, "mask_id", model.mask_index)
        samples = torch.full(
            (request.num_samples, model.num_tokens),
            mask_id,
            dtype=torch.long,
            device=model.device,
        )
        samples[:, 0] = model.tokenizer.bos_token_id
        generated = samples[:, :1]
        key_positions = torch.zeros_like(generated)
        noise = (
            torch.distributions.Gumbel(0, 1)
            .sample((request.num_samples, model.num_tokens - 1, model.vocab_size))
            .to(model.device)
        )
        if self.use_float64:
            noise = noise.to(torch.float64)
        tokens_processed = 0
        for step in range(model.num_tokens - 1):
            query_positions = order[:, : step + 1] + 1
            logits = model.permuted_logits(generated, key_positions, query_positions)
            log_probs = (logits[:, -1] / self.temperature).log_softmax(-1)
            token = (log_probs + noise[:, step]).argmax(-1)
            target_position = query_positions[:, -1:]
            samples.scatter_(1, target_position, token[:, None])
            generated = torch.cat((generated, token[:, None]), dim=1)
            key_positions = torch.cat((key_positions, target_position), dim=1)
            tokens_processed += generated.shape[1] - 1
        return SamplingResult(
            samples,
            {
                "nfe": model.num_tokens - 1,
                "tokens_processed": tokens_processed,
                "cache_used": False,
                "order": order,
            },
        )


@sampler_registry.register("permuted_ar_uncached")
def _create_uncached_permuted_ar(config):
    return UncachedPermutedAutoregressiveSampler(
        use_float64=config.sampling.use_float64,
        temperature=config.algo.get("temp", 1.0),
    )

from __future__ import annotations

import torch

from .base import Sampler, SamplerSpec, SamplingRequest, SamplingResult
from .registry import sampler_registry


def generation_order(model, request):
    supplied = request.options.get("order")
    length = model.num_tokens - 1
    if supplied is not None:
        order = supplied.to(device=model.device, dtype=torch.long)
        if order.shape != (request.num_samples, length):
            raise ValueError(
                "supplied order must have shape [num_samples, num_tokens - 1]"
            )
    else:
        mode = request.options.get("order_mode", "random")
        if mode == "sequential":
            order = torch.arange(length, device=model.device).expand(
                request.num_samples, -1
            )
        elif mode == "random":
            order = torch.rand(
                request.num_samples, length, device=model.device
            ).argsort(-1)
        elif mode == "block":
            block_size = int(request.options.get("block_size", -1))
            if block_size <= 0:
                raise ValueError("block order requires a positive block_size")
            blocks = []
            for start in range(0, length, block_size):
                width = min(block_size, length - start)
                blocks.append(
                    torch.rand(request.num_samples, width, device=model.device).argsort(
                        -1
                    )
                    + start
                )
            order = torch.cat(blocks, dim=1)
        else:
            raise ValueError(f"unknown permutation order mode: {mode}")
    expected = torch.arange(length, device=model.device).expand_as(order)
    if not torch.equal(order.sort(-1).values, expected):
        raise ValueError("each generation order row must be a permutation")
    return order


class PermutedAutoregressiveSampler(Sampler):
    spec = SamplerSpec(
        name="permuted_ar",
        compatible_families=frozenset({"permuted_autoregressive"}),
        required_capabilities=frozenset({"cached_permuted_logits"}),
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
        noise = (
            torch.distributions.Gumbel(0, 1)
            .sample((request.num_samples, model.num_tokens - 1, model.vocab_size))
            .to(model.device)
        )
        if self.use_float64:
            noise = noise.to(torch.float64)

        cache = None
        current_tokens = samples[:, :1]
        key_positions = torch.zeros_like(current_tokens)
        for step in range(model.num_tokens - 1):
            query_positions = order[:, step : step + 1] + 1
            logits, cache = model.permuted_logits_cached(
                current_tokens, key_positions, query_positions, cache=cache
            )
            log_probs = (logits[:, -1] / self.temperature).log_softmax(-1)
            token = (log_probs + noise[:, step]).argmax(-1)
            target_positions = query_positions
            samples.scatter_(1, target_positions, token[:, None])
            current_tokens = token[:, None]
            key_positions = target_positions
        return SamplingResult(
            samples,
            {
                "nfe": model.num_tokens - 1,
                "tokens_processed": model.num_tokens - 1,
                "cache_used": True,
                "order": order,
            },
        )


@sampler_registry.register("permuted_ar")
def _create_permuted_ar(config):
    return PermutedAutoregressiveSampler(
        use_float64=config.sampling.use_float64,
        temperature=config.algo.get("temp", 1.0),
    )

"""Cached arbitrary-order decoding; EOS never ends spatial filling early."""

import torch
from .autoregressive import AutoregressiveSampler
from core.algorithms.permuted_autoregressive import PermutedAutoregressive


class PermutedAutoregressiveSampler(AutoregressiveSampler):
    def __init__(self, order, **kwargs):
        super().__init__(**kwargs)
        self.order = order

    @torch.no_grad()
    def sample(self, algo, prompt_tokens, prompt_mask):
        if not isinstance(algo, PermutedAutoregressive):
            raise TypeError("permuted sampler requires PermutedAutoregressive")
        if (
            prompt_tokens.ndim != 2
            or prompt_tokens.shape != prompt_mask.shape
            or not all(prompt_tokens.shape)
        ):
            raise ValueError("expected matching nonempty [batch, length] prompts")
        fixed = prompt_mask.bool()
        if fixed[:, 1:].any() and not algo.condition_on_observed:
            raise ValueError(
                "prompt conditioning requires condition_on_observed training"
            )
        if (~fixed[:, 0]).any() and self.bos_token_id is None:
            raise ValueError(
                "unconditional generation requires bos_token_id or task tokenizer metadata"
            )
        result = torch.where(
            fixed, prompt_tokens, 0 if self.pad_token_id is None else self.pad_token_id
        ).clone()
        if self.bos_token_id is not None:
            result[:, 0] = torch.where(fixed[:, 0], result[:, 0], self.bos_token_id)
        order = self.order.sample(
            len(result), result.shape[1] - 1, device=result.device
        )
        order = self.order.observed_first(order, fixed[:, 1:]) + 1
        key_positions = torch.cat((torch.zeros_like(result[:, :1]), order[:, :-1]), 1)
        cache = (
            algo.model.allocate_kv_cache(
                len(result), result.shape[1], device=result.device
            )
            if self.use_cache
            else None
        )
        processed = 0
        for step in range(order.shape[1]):
            left = step if self.use_cache else 0
            keys, queries = key_positions[:, left : step + 1], order[:, left : step + 1]
            inputs = result.gather(1, keys)
            prediction = algo.sampling_prediction(
                inputs, cache=cache, key_position_ids=keys, query_position_ids=queries
            )[:, -1]
            probabilities = algo.predictor.probabilities(
                prediction.float(), apply_commitment=True
            )
            sampled = torch.multinomial(probabilities, 1)
            positions = order[:, step : step + 1]
            selected = torch.where(
                fixed.gather(1, positions), result.gather(1, positions), sampled
            )
            result.scatter_(1, positions, selected)
            processed += inputs.numel()
        return result, {
            "nfe": order.shape[1],
            "nfe_per_sample": [order.shape[1]] * len(result),
            "tokens_processed": processed,
            "cache_used": self.use_cache,
            "order": (order - 1).cpu().tolist(),
        }

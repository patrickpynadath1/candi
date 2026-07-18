from __future__ import annotations

import torch

from .ar import AR
from .registry import algo_registry
from .trainer_base import Loss


def sample_generation_order(
    batch_size, length, *, device, mode="random", block_size=-1
):
    """Return output indices in generation order, relative to the shifted target."""
    sequential = torch.arange(length, device=device).expand(batch_size, -1)
    if mode == "sequential":
        return sequential
    if mode == "random":
        return torch.rand(batch_size, length, device=device).argsort(dim=-1)
    if mode != "block":
        raise ValueError(f"unknown permutation order mode: {mode}")
    if block_size is None or block_size <= 0:
        raise ValueError("block order requires a positive block_size")
    blocks = []
    for start in range(0, length, block_size):
        width = min(block_size, length - start)
        blocks.append(torch.rand(batch_size, width, device=device).argsort(-1) + start)
    return torch.cat(blocks, dim=1)


def permute_autoregressive_batch(tokens, valid_tokens, order):
    """Build BOS-shifted inputs and decoupled positions for an output order."""
    targets = tokens[:, 1:]
    if order.shape != targets.shape:
        raise ValueError("generation order must match the shifted target shape")
    ordered_targets = targets.gather(1, order)
    inputs = torch.cat((tokens[:, :1], ordered_targets[:, :-1]), dim=1)
    key_positions = torch.cat(
        (
            torch.zeros(order.shape[0], 1, dtype=torch.long, device=order.device),
            order[:, :-1] + 1,
        ),
        dim=1,
    )
    query_positions = order + 1
    ordered_valid = valid_tokens[:, 1:].gather(1, order)
    return inputs, ordered_targets, ordered_valid, key_positions, query_positions


@algo_registry.register("mdlm_perm")
class MDLMPerm(AR):
    """Permutation-trained causal model with decoupled query/key RoPE."""

    default_sampler = "permuted_ar"
    supported_samplers = frozenset({"permuted_ar", "permuted_ar_uncached"})
    sampling_family = "permuted_autoregressive"
    sampling_capabilities = frozenset(
        {
            "permuted_logits",
            "cached_permuted_logits",
        }
    )

    def _order_options(self):
        mode = self.config.algo.get("order", "random")
        ar_probability = self.config.algo.get("ar_probability", 0.0)
        block_size = self.config.algo.get("block_size", -1)
        return mode, ar_probability, block_size

    def _loss(self, x0, valid_tokens, current_accumulation_step=None, train_mode=False):
        del current_accumulation_step, train_mode
        mode, ar_probability, block_size = self._order_options()
        order = sample_generation_order(
            x0.shape[0],
            x0.shape[1] - 1,
            device=x0.device,
            mode=mode,
            block_size=block_size,
        )
        if ar_probability:
            sequential = torch.arange(order.shape[1], device=x0.device).expand_as(order)
            use_ar = torch.rand(x0.shape[0], device=x0.device) < ar_probability
            order = torch.where(use_ar[:, None], sequential, order)
        inputs, targets, output_valid, key_positions, query_positions = (
            permute_autoregressive_batch(x0, valid_tokens, order)
        )
        logits = self.permuted_logits(inputs, key_positions, query_positions)
        nll = -logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
        nlls = (nll * output_valid).sum()
        num_tokens = output_valid.sum()
        return Loss(
            loss=nlls / num_tokens,
            nlls=nlls,
            prior_loss=0.0,
            num_tokens=num_tokens,
        )

    def permuted_logits(self, tokens, key_position_ids, query_position_ids):
        logits, _ = self.backbone.forward_positioned(
            tokens,
            key_position_ids=key_position_ids,
            query_position_ids=query_position_ids,
        )
        logits[:, :, self.mask_index] = self.neg_infinity
        return logits

    def permuted_logits_cached(
        self, tokens, key_position_ids, query_position_ids, cache=None
    ):
        if cache is None:
            cache = self.backbone.allocate_kv_cache(
                tokens.shape[0], self.num_tokens, device=tokens.device
            )
        logits, cache = self.backbone.forward_positioned(
            tokens,
            key_position_ids=key_position_ids,
            query_position_ids=query_position_ids,
            cache=cache,
        )
        logits[:, :, self.mask_index] = self.neg_infinity
        return logits, cache

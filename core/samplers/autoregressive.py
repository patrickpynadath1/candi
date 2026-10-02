"""Prefix-prefill and single-token decoding, with optional uncached reference."""

import torch
from core.algorithms.autoregressive import Autoregressive


class AutoregressiveSampler:
    def __init__(
        self, use_cache=True, bos_token_id=None, eos_token_id=None, pad_token_id=None
    ):
        self.use_cache = use_cache
        self.bos_token_id, self.eos_token_id, self.pad_token_id = (
            bos_token_id,
            eos_token_id,
            pad_token_id,
        )

    def configure_special_tokens(
        self, *, bos_token_id=None, eos_token_id=None, pad_token_id=None
    ):
        for name, value in (
            ("bos_token_id", bos_token_id),
            ("eos_token_id", eos_token_id),
            ("pad_token_id", pad_token_id),
        ):
            if getattr(self, name) is None:
                setattr(self, name, value)

    @torch.no_grad()
    def sample(self, algo, prompt_tokens, prompt_mask):
        if not isinstance(algo, Autoregressive):
            raise TypeError("autoregressive sampler requires Autoregressive algorithm")
        if (
            prompt_tokens.ndim != 2
            or prompt_tokens.shape != prompt_mask.shape
            or not all(prompt_tokens.shape)
        ):
            raise ValueError("expected matching nonempty [batch, length] prompts")
        fixed = prompt_mask.bool()
        lengths = fixed.sum(-1)
        positions = torch.arange(prompt_tokens.shape[1], device=prompt_tokens.device)
        if not torch.equal(fixed, positions[None] < lengths[:, None]):
            raise ValueError(
                "autoregressive conditioning requires contiguous prefix masks"
            )
        pad = 0 if self.pad_token_id is None else self.pad_token_id
        result = torch.where(fixed, prompt_tokens, pad).clone()
        nfe = torch.zeros(len(result), dtype=torch.long, device=result.device)
        processed = 0
        model_calls = 0
        for prefix in lengths.unique().tolist():
            indices = (lengths == prefix).nonzero(as_tuple=True)[0]
            state = result[indices].clone()
            if prefix == 0:
                if self.bos_token_id is None:
                    raise ValueError(
                        "unconditional AR generation requires bos_token_id or task tokenizer metadata"
                    )
                state[:, 0] = self.bos_token_id
            start = max(prefix, 1)
            cache = (
                algo.model.allocate_kv_cache(
                    len(state), state.shape[1], device=state.device
                )
                if self.use_cache
                else None
            )
            finished = torch.zeros(len(state), dtype=torch.bool, device=state.device)
            counts = torch.zeros(len(state), dtype=torch.long, device=state.device)
            for index in range(start, state.shape[1]):
                inputs = (
                    state[:, :index]
                    if cache is None or cache.length == 0
                    else state[:, index - 1 : index]
                )
                prediction = algo.sampling_prediction(inputs, cache=cache)[:, -1]
                probabilities = algo.predictor.probabilities(
                    prediction.float(), apply_commitment=True
                )
                next_tokens = torch.multinomial(probabilities, 1).squeeze(-1)
                state[:, index] = torch.where(finished, pad, next_tokens)
                counts += ~finished
                processed += inputs.numel()
                model_calls += 1
                if self.eos_token_id is not None:
                    finished |= next_tokens.eq(self.eos_token_id)
                    if finished.all():
                        break
            result[indices] = state
            nfe[indices] = counts
        return result, {
            "nfe": model_calls,
            "nfe_per_sample": nfe.cpu().tolist(),
            "tokens_processed": processed,
            "cache_used": self.use_cache,
        }

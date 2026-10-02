"""Sequential block orchestration around an existing diffusion sampler."""

import torch


class _ActiveBlock:
    """Length override while preserving every mathematical algorithm component."""
    def __init__(self, algorithm, length):
        self.algorithm, self.length = algorithm, length

    def __getattr__(self, name):
        return getattr(self.algorithm, name)


class BlockSampler:
    def __init__(self, sampler, block_size, cache_clean_prefix=True):
        if not hasattr(sampler, "sample"):
            raise TypeError("BlockSampler requires an inner sampler with sample()")
        if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        self.sampler, self.block_size = sampler, block_size
        self.cache_clean_prefix = bool(cache_clean_prefix)

    @torch.no_grad()
    def sample(self, algo, prompt_tokens, prompt_mask):
        if prompt_tokens.ndim != 2 or prompt_tokens.shape != prompt_mask.shape:
            raise ValueError("prompt tensors must have matching [batch, length] shapes")
        result = prompt_tokens.clone()
        known = prompt_mask.bool().clone()
        length = result.shape[1]
        total = [0] * len(result)
        for start in range(0, length, self.block_size):
            stop = min(start + self.block_size, length)
            active_unknown = ~known[:, start:stop]
            if not active_unknown.any():
                continue
            block_tokens = result[:, start:stop]
            block_prompt = known[:, start:stop]
            prefix = result[:, :start]
            cache = {} if self.cache_clean_prefix else None
            with algo.model.clean_context(prefix, offset=start, cache=cache):
                block_result, metadata = self.sampler.sample(
                    _ActiveBlock(algo.algorithm, stop - start),
                    block_tokens, block_prompt
                )
            result[:, start:stop] = torch.where(
                block_prompt, block_tokens, block_result
            )
            known[:, start:stop] = True
            counts = metadata.get("nfe_per_sample")
            if counts is None:
                if "nfe" not in metadata:
                    raise ValueError("inner sampler must return NFE metadata")
                counts = [metadata["nfe"]] * len(result)
            if len(counts) != len(result):
                raise ValueError("inner sampler per-sample NFE has wrong length")
            total = [a + int(b) for a, b in zip(total, counts)]
        return result, {"nfe": max(total, default=0), "nfe_per_sample": total}

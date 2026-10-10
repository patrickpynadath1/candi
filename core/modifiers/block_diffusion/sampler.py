"""Sequential block orchestration around an existing diffusion sampler."""

import torch


class _ActiveBlock:
    """Length override while preserving every mathematical algorithm component."""

    def __init__(self, algorithm, length, max_length=None):
        self._wrapped, self.length = algorithm, length
        self.max_length = length if max_length is None else max_length

    @property
    def algorithm(self):
        # InsertionSampler uses this for assembly-level synchronization while
        # all structural methods still resolve on VariableLengthDiffusion.
        return self._wrapped.algorithm

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


class BlockSampler:
    def __init__(self, sampler, block_size, cache_clean_prefix=True, max_length=None):
        if not hasattr(sampler, "sample"):
            raise TypeError("BlockSampler requires an inner sampler with sample()")
        if (
            not isinstance(block_size, int)
            or isinstance(block_size, bool)
            or block_size < 1
        ):
            raise ValueError("block_size must be a positive integer")
        self.sampler, self.block_size = sampler, block_size
        self.cache_clean_prefix = bool(cache_clean_prefix)
        self.max_length = block_size if max_length is None else max_length

    @torch.no_grad()
    def sample(self, algo, prompt_tokens, prompt_mask):
        if prompt_tokens.ndim != 2 or prompt_tokens.shape != prompt_mask.shape:
            raise ValueError("prompt tensors must have matching [batch, length] shapes")
        if getattr(algo.algorithm, "supports_variable_length", False):
            return self._sample_variable(algo, prompt_tokens, prompt_mask)
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
                    block_tokens,
                    block_prompt,
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

    def _sample_variable(self, algo, prompt_tokens, prompt_mask):
        """Complete logical blocks independently, one example at a time.

        Per-example execution keeps every clean prefix free of padding. A block's
        output is frozen before the next block starts, so shifting positions can
        never move an edit into another logical block.
        """
        rows, totals, all_lengths, all_inserted, all_truncated = [], [], [], [], []
        per_block = []
        source_length = prompt_tokens.shape[1]
        for row in range(len(prompt_tokens)):
            completed = prompt_tokens.new_empty((1, 0))
            nfe = 0
            block_lengths, inserted, truncated = [], 0, False
            for start in range(0, source_length, self.block_size):
                stop = min(start + self.block_size, source_length)
                block_tokens = prompt_tokens[row : row + 1, start:stop]
                block_prompt = prompt_mask[row : row + 1, start:stop].bool()
                if block_prompt.all():
                    completed = torch.cat([completed, block_tokens], 1)
                    block_lengths.append(stop - start)
                    continue
                # Prefix caches are intentionally not shared across variable
                # blocks: a completed edit changes all following positions.
                with algo.model.clean_context(
                    completed, offset=completed.shape[1], cache=None
                ):
                    output, metadata = self.sampler.sample(
                        _ActiveBlock(
                            algo.algorithm,
                            stop - start,
                            max_length=self.max_length,
                        ),
                        block_tokens,
                        block_prompt,
                    )
                length = int(metadata["lengths"][0])
                completed = torch.cat([completed, output[:, :length]], 1)
                block_lengths.append(length)
                nfe += int(metadata.get("nfe_per_sample", [metadata["nfe"]])[0])
                inserted += int(metadata.get("inserted_tokens", [0])[0])
                truncated |= bool(metadata.get("truncated", [False])[0])
            rows.append(completed.squeeze(0))
            totals.append(nfe)
            all_lengths.append(completed.shape[1])
            all_inserted.append(inserted)
            all_truncated.append(truncated)
            per_block.append(block_lengths)
        width = max(all_lengths, default=0)
        result = prompt_tokens.new_full((len(rows), width), self.sampler.pad_token_id)
        for row, value in enumerate(rows):
            result[row, : len(value)] = value
        return result, {
            "nfe": max(totals, default=0),
            "nfe_per_sample": totals,
            "lengths": all_lengths,
            "block_lengths": per_block,
            "inserted_tokens": all_inserted,
            "truncated": all_truncated,
        }

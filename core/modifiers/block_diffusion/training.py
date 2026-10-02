"""Algorithm modifier adding block conditioning and optional clean AR loss."""

import torch
from torch import nn
from torch.nn import functional as F
from .model import BlockDiffusionModel
from core.sequences import SequenceLayout


class BlockDiffusion(nn.Module):
    def __init__(
        self,
        algorithm,
        block_size,
        clean_loss="none",
        clean_loss_weight=1.0,
        cache_clean_prefix=True,
    ):
        super().__init__()
        if clean_loss not in {"none", "autoregressive"}:
            raise ValueError("clean_loss must be none or autoregressive")
        if clean_loss_weight < 0:
            raise ValueError("clean_loss_weight must be nonnegative")
        if type(algorithm).__name__ == "RePlaid":
            raise TypeError(
                "block diffusion does not support RePlaid's specialized gradient "
                "routing and optimizer parameter groups; use MDLM or GaussianDiffusion"
            )
        self.algorithm = algorithm
        clean_encoder = getattr(
            getattr(algorithm, "representation", None), "encode", None
        )
        self.algorithm.model = BlockDiffusionModel(
            algorithm.model, block_size, clean_encoder=clean_encoder
        )
        self.block_size, self.clean_loss = block_size, clean_loss
        self.clean_loss_weight = float(clean_loss_weight)
        self.cache_clean_prefix = bool(cache_clean_prefix)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.algorithm, name)

    @property
    def model(self):
        return self.algorithm.model

    def loss(self, tokens, loss_mask, *args, **kwargs):
        if getattr(self.algorithm, "supports_sequence_layout", False):
            if "sequence_layout" in kwargs:
                raise ValueError("block diffusion owns sequence_layout construction")
            kwargs["sequence_layout"] = SequenceLayout.fixed_blocks(
                len(tokens), tokens.shape[1], self.block_size, tokens.device
            )
        self.model.last_clean_logits = None
        try:
            with self.model.clean_context(tokens):
                result = self.algorithm.loss(tokens, loss_mask, *args, **kwargs)
            logits = self.model.last_clean_logits
        finally:
            self.model.last_clean_logits = None
        if self.clean_loss == "none" or self.clean_loss_weight == 0:
            return result
        valid = loss_mask[:, 1:].bool()
        count = valid.sum()
        if count == 0:
            return result
        per_token = F.cross_entropy(
            logits[:, :-1].transpose(1, 2), tokens[:, 1:], reduction="none"
        )
        ar_sum = (per_token * valid.to(per_token)).sum()
        ar_loss = ar_sum / count
        result["diffusion_loss"] = result["loss"]
        result["clean_loss"] = ar_loss
        result["loss"] = result["loss"] + self.clean_loss_weight * ar_loss
        result["loss_sum"] = result["loss"].detach() * result["num_tokens"]
        return result

    @torch.no_grad()
    def generate(self, sampler, prompt_tokens, prompt_mask):
        from .sampler import BlockSampler

        if not isinstance(sampler, BlockSampler):
            sampler = BlockSampler(
                sampler, self.block_size, cache_clean_prefix=self.cache_clean_prefix
            )
        was_training = self.training
        self.eval()
        try:
            return sampler.sample(self, prompt_tokens, prompt_mask)
        finally:
            self.train(was_training)

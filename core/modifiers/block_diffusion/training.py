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
        max_length=None,
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
        variable_length = getattr(algorithm, "supports_variable_length", False)
        if max_length is not None and (
            not isinstance(max_length, int)
            or isinstance(max_length, bool)
            or max_length < 1
        ):
            raise ValueError("max_length must be a positive integer")
        if variable_length and clean_loss != "none":
            raise TypeError(
                "clean autoregressive loss is not defined for variable active blocks"
            )
        self.algorithm = algorithm
        clean_encoder = getattr(
            getattr(algorithm, "representation", None), "encode", None
        )
        wrapped_model = BlockDiffusionModel(
            algorithm.model, block_size, clean_encoder=clean_encoder
        )
        # VariableLengthDiffusion exposes its assembly model through a read-only
        # property. Install the visibility wrapper on that owning assembly.
        model_owner = algorithm.algorithm if variable_length else algorithm
        model_owner.model = wrapped_model
        self.block_size, self.clean_loss = block_size, clean_loss
        self.max_length = block_size if max_length is None else max_length
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
        if getattr(self.algorithm, "supports_variable_length", False):
            return self._variable_length_loss(tokens, loss_mask, *args, **kwargs)
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

    def _variable_length_loss(self, tokens, loss_mask, *args, **kwargs):
        """Corrupt all logical blocks and train them in one fused model pass."""
        if tokens.ndim != 2 or loss_mask.shape != tokens.shape or not tokens.shape[1]:
            raise ValueError("tokens and loss_mask must have matching nonempty shapes")
        original_valid = kwargs.get("valid_mask")
        if original_valid is not None and not original_valid.bool().all():
            raise TypeError("block insertion training requires packed valid sequences")
        batch, length = tokens.shape
        blocks = (length + self.block_size - 1) // self.block_size
        padding = blocks * self.block_size - length

        def split(value, fill=0):
            return F.pad(value, (0, padding), value=fill).reshape(
                batch * blocks, self.block_size
            )

        active_kwargs = dict(kwargs)
        valid = split(torch.ones_like(tokens, dtype=torch.bool), False)
        active_kwargs["valid_mask"] = valid
        if active_kwargs.get("corruption_mask") is not None:
            active_kwargs["corruption_mask"] = split(
                active_kwargs["corruption_mask"], False
            )
        # Keep antithetic/DDP/accumulation partitioning in original-example units.
        # Every block of an example shares its sampled global noise time.
        active_kwargs["time_group_size"] = blocks
        self.model.last_clean_logits = None
        try:
            with self.model.parallel_clean_context(tokens, blocks):
                return self.algorithm.loss(
                    split(tokens), split(loss_mask), *args, **active_kwargs
                )
        finally:
            self.model.last_clean_logits = None

    @torch.no_grad()
    def generate(self, sampler, prompt_tokens, prompt_mask):
        from .sampler import BlockSampler

        if not isinstance(sampler, BlockSampler):
            sampler = BlockSampler(
                sampler,
                self.block_size,
                cache_clean_prefix=self.cache_clean_prefix,
                max_length=self.max_length,
            )
        was_training = self.training
        self.eval()
        try:
            return sampler.sample(self, prompt_tokens, prompt_mask)
        finally:
            self.train(was_training)

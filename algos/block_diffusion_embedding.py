"""Rolling block diffusion with CANDI Gaussian embedding schedules."""

import torch

from .block_diffusion import BlockDiffusion
from .rank_schedules import SphereRankSchedule
from .registry import algo_registry


@algo_registry.register("block_diffusion_embedding")
class BlockDiffusionEmbedding(BlockDiffusion):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.interpolation_schedule = config.algo.interpolation_schedule
        if self.interpolation_schedule not in {"linear", "linear_rank", "geometric_rank"}:
            raise ValueError("Expected linear, linear_rank, or geometric_rank interpolation")
        if (config.noise.eps != 0 or config.noise.get("alpha_min") is not None
                or config.noise.get("alpha_max") is not None):
            raise ValueError("Block embedding schedules require untruncated noise with eps=0")
        if config.noise.get("adaptive", False) and config.noise.get("adaptive_coordinate") != "physical":
            raise ValueError("Adaptive block schedules require the physical coordinate")
        if self.interpolation_schedule != "linear":
            if self.embedding_scale != 1:
                raise ValueError("Embedding rank calibration requires embedding_scale=1")
            width = float(config.algo.rank_endpoint_width)
            if not 0 < width < .5:
                raise ValueError("Rolling rank schedules require 0 < rank_endpoint_width < .5")
            self.rank_schedule = SphereRankSchedule(
                config.model.hidden_size, config.algo.min_percentile,
                config.algo.max_percentile, width,
            )

    def interpolation(self, times):
        # Apply the same monotone clock to every position; never resample or
        # sort positions here. Sampler times remain in this clock coordinate.
        progress = 1 - self.noise.alpha_t(times)
        if self.interpolation_schedule == "linear":
            return progress
        coupling = "linear" if self.interpolation_schedule == "linear_rank" else "geometric"
        return self.rank_schedule(progress, coupling=coupling)

    def _record_block_time_loss(self, times, losses, valid_tokens, train_mode):
        if not train_mode or not self.config.noise.get("adaptive", False):
            return
        # Each position has its own clock. Fit raw noisy next-token CE, not
        # the auxiliary clean-stream loss or a sequence-averaged clock.
        # All ranks must participate, including ranks with no valid tokens.
        valid = valid_tokens.bool() & torch.isfinite(losses)
        with torch.no_grad():
            self.noise.record_time_loss_pair(
                times[valid].detach(), losses[valid].detach(), int(self.global_step)
            )
        if getattr(self, "_trainer", None) is not None:
            self._log_adaptive_schedule()

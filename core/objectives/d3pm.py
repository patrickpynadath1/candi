"""Absorbing D3PM variational objective including clean-input reconstruction."""

import torch


class D3PMAbsorbingObjective:
    def __init__(self, mask_index, steps=1000):
        self.mask_index, self.steps = mask_index, steps

    def __call__(self, prediction, state, targets, keep, reconstruction):
        dt = 1 / self.steps
        time = (1 - keep[:, None]).clamp(0, 1 - 1e-4)
        if torch.any(time <= 0):
            raise ValueError("D3PM loss requires positive corruption time")
        clean_log_probability = prediction.gather(-1, targets[..., None]).squeeze(-1)
        mask_probability = prediction[..., self.mask_index].exp()
        log_normalizer = torch.log1p(keep[:, None] * mask_probability / time)
        # Reference alpha_s is (time-dt), so its ratio cancels analytically.
        # This avoids 0/0 at the first step without changing interior values.
        weight = dt / time
        bound = weight * (log_normalizer - clean_log_probability) + (1 - weight) * (
            log_normalizer - torch.log1p(mask_probability)
        )
        reconstruction_ce = -reconstruction.gather(-1, targets[..., None]).squeeze(-1)
        return reconstruction_ce + self.steps * bound * state.eq(self.mask_index)

"""Exact masking with CANDI's data-vocabulary posterior transition."""

import torch
from .absorbing import AbsorbingProcess
from core.samplers.utils import sample_categorical


class CANDIMaskingProcess(AbsorbingProcess):
    def reverse(self, state, posterior, keep_t, keep_s):
        if torch.any((keep_t < 0) | (keep_s > 1) | (keep_s < keep_t)):
            raise ValueError("reverse transition requires 0 <= keep_t <= keep_s <= 1")
        data = posterior.double() * (keep_s - keep_t)[:, None, None]
        mask = (1 - keep_s)[:, None, None].expand(*state.shape, 1)
        proposed = sample_categorical(torch.cat([data, mask], -1))
        return torch.where(state.eq(self.mask_index), proposed, state)

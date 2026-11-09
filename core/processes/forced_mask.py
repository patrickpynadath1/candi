"""Forced-target absorbing proposal for low-variance conditional estimators."""
import torch


class ForcedTargetMask:
    def select_target(self, valid):
        if valid.dtype != torch.bool:
            raise ValueError("forced-target proposal requires a boolean mask")
        count = valid.sum(-1)
        weights = valid.float() + (count == 0)[:, None].float()
        target = torch.multinomial(weights, 1).squeeze(-1)
        return target

    def __call__(self, valid, probability, target=None):
        count = valid.sum(-1)
        if target is None:
            target = self.select_target(valid)
        masked = (torch.rand(valid.shape, device=valid.device) < probability[:, None]) & valid
        masked[torch.arange(len(valid), device=valid.device), target] = count > 0
        return masked

"""Reconstruction/diffusion batch allocation and stratified continuous times."""

from dataclasses import dataclass
import torch
from torch import nn
from core.objectives.reductions import masked_example_mean


@dataclass(frozen=True)
class TimeAllocation:
    time: torch.Tensor
    reconstruction_count: int


class AdaptiveTimeAllocation(nn.Module):
    def __init__(self, momentum=0.99):
        super().__init__()
        if not 0 <= momentum < 1:
            raise ValueError("allocation momentum must be in [0, 1)")
        self.momentum = momentum
        self.register_buffer("recon_std_ema", torch.tensor(1.0))
        self.register_buffer("diff_std_ema", torch.tensor(1.0))
        for name in ("recon", "diff"):
            self.register_buffer(f"{name}_mean_ema", torch.tensor(0.0))
            self.register_buffer(f"{name}_second_ema", torch.tensor(1.0))

    def sample(self, count, device, **context):
        if count < 1:
            raise ValueError("allocation requires a nonempty batch")
        ratio = self.recon_std_ema / (self.recon_std_ema + self.diff_std_ema).clamp_min(
            1e-12
        )
        recon = int((count * ratio).round().clamp(1, count - 1)) if count > 1 else 0
        diffusion = count - recon
        time = torch.zeros(count, device=device)
        time[recon:] = (
            torch.rand((), device=device)
            + torch.arange(diffusion, device=device) / diffusion
        ) % 1
        return TimeAllocation(time, recon)

    @torch.no_grad()
    def observe(self, reconstruction, diffusion, reconstruction_count, loss_mask=None):
        if not 0 < reconstruction_count < len(diffusion):
            return
        values = {
            "recon": masked_example_mean(reconstruction, loss_mask)[
                :reconstruction_count
            ],
            "diff": masked_example_mean(diffusion, loss_mask)[reconstruction_count:],
        }
        for name, samples in values.items():
            mean = getattr(self, f"{name}_mean_ema")
            second = getattr(self, f"{name}_second_ema")
            mean.lerp_(samples.mean().to(mean), 1 - self.momentum)
            second.lerp_(samples.square().mean().to(second), 1 - self.momentum)
            getattr(self, f"{name}_std_ema").copy_(
                (second - mean.square()).clamp_min(1e-12).sqrt()
            )

"""Teacher target construction, independent of losses and optimizer steps."""

import torch
from torch import nn
from core.processes.absorbing import AbsorbingProcess


class UnmaskingRollout(nn.Module):
    def __init__(self, steps=2, step_size=1 / 128):
        super().__init__()
        if steps < 1 or int(steps) != steps or not 0 < step_size <= 1:
            raise ValueError("invalid rollout steps or step_size")
        self.steps = int(steps)
        self.register_buffer("step_size", torch.tensor(float(step_size)))

    @torch.no_grad()
    def forward(self, models, state):
        teacher = models["teacher"]
        if len(teacher.views) != 1 or not isinstance(
            teacher.views[0].process, AbsorbingProcess
        ):
            raise ValueError("unmasking targets require a single absorbing view")
        process = teacher.views[0].process
        tokens = state["states"][0].clone()
        time = state["time"].clone()
        targets = None
        for index in range(self.steps):
            points = teacher.points(time)
            logp = teacher.sampling_prediction((tokens,), points, time)
            if targets is None:
                targets = logp.clone()
            destination = (time - self.step_size).clamp_min(1e-5)
            next_points = teacher.points(destination)
            next_tokens = process.reverse(
                tokens,
                logp.exp(),
                points[0].coefficients[0],
                next_points[0].coefficients[0],
            )
            unresolved = tokens.eq(process.mask_index)
            revealed = unresolved & next_tokens.ne(process.mask_index)
            take = unresolved if index == self.steps - 1 else revealed
            targets = torch.where(take[..., None], logp, targets)
            tokens, time = next_tokens, destination
        return targets

"""Training-state constructors reuse the information path owned by the recipe."""

import torch
from torch import nn
from core.processes.absorbing import AbsorbingProcess


class CorruptData(nn.Module):
    def __init__(self, only_corrupted=False, role="student", minimum_time=1e-5):
        super().__init__()
        self.only_corrupted = only_corrupted
        self.role, self.minimum_time = role, minimum_time

    def forward(self, models, batch, step):
        student = models[self.role]
        tokens = batch["input_ids"]
        corruption = batch.get("corruption_mask", batch["loss_mask"]).bool()
        time = self.minimum_time + (1 - self.minimum_time) * torch.rand(
            len(tokens), device=tokens.device
        )
        time = time.clamp_max(1 - 1e-5)
        points = student.points(time)
        states = tuple(
            view.process.corrupt_state(
                view.representation.encode(tokens), p.coefficients, corruption
            )
            for view, p in zip(student.views, points)
        )
        mask = batch["loss_mask"].bool()
        if self.only_corrupted:
            if len(student.views) != 1 or not isinstance(
                student.views[0].process, AbsorbingProcess
            ):
                raise ValueError("only_corrupted requires a single absorbing view")
            mask = mask & states[0].eq(student.views[0].process.mask_index)
        return dict(states=states, points=points, time=time, tokens=tokens, mask=mask)


class PosteriorPrediction(nn.Module):
    def forward(self, models, state):
        return models["student"].training_prediction(
            state["states"], state["points"], state["time"]
        )


class DataTargets(nn.Module):
    def forward(self, models, state):
        return state["tokens"]

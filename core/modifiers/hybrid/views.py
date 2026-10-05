"""Passive composition records. View behavior belongs to its components."""

from dataclasses import dataclass
import torch
from torch import nn
from .time_maps import IdentityTimeMap


@dataclass(frozen=True)
class ViewPoint:
    time: torch.Tensor
    coefficients: object


class DiffusionView(nn.Module):
    def __init__(
        self, process, schedule, representation, update, time_map=None, name=None
    ):
        super().__init__()
        self.process = process
        self.schedule = schedule
        self.representation = representation
        self.update = update
        self.time_map = time_map or IdentityTimeMap()
        self.name = name or type(process).__name__
        requirements = getattr(schedule, "representation_requirements", None)
        if requirements is not None:
            requirements.validate(representation.spec, type(schedule).__name__)


def point(view, time, *, training=False, corruption_mask=None, sequence_layout=None):
    local = view.time_map(time)
    if training:
        local = getattr(view.schedule, "training_time", lambda t: t)(local)
    if hasattr(view.schedule, "for_layout"):
        from core.sequences import resolve_sequence_layout

        if corruption_mask is None:
            raise ValueError("a sequence-aware schedule requires a corruption mask")
        layout = resolve_sequence_layout(corruption_mask, sequence_layout)
        coefficients = view.schedule.for_layout(
            local, corruption_mask, view.representation.spec.dimension, layout
        )
    else:
        coefficients = view.schedule(local)
    return ViewPoint(local, coefficients)


def clamp(view, state, tokens, mask):
    clean = view.representation.clean_state(torch.where(mask.bool(), tokens, 0))
    while mask.ndim < state.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask.bool(), clean, state)


def prior(view, tokens, local_point, prompt_mask):
    clean = view.representation.encode(tokens)
    state = view.process.prior_state(clean, local_point.coefficients)
    return clamp(view, state, tokens, prompt_mask)

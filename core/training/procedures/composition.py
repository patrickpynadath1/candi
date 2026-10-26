"""Training phases compose states, predictions, targets, losses and reductions."""

import copy
from contextlib import contextmanager
import torch
from torch import nn


class Phase(nn.Module):
    def __init__(
        self, states, prediction, targets, objective, reduction, role="student"
    ):
        super().__init__()
        self.states, self.prediction, self.targets = states, prediction, targets
        self.objective, self.reduction, self.role = objective, reduction, role

    def forward(self, models, batch, step):
        state = self.states(models, batch, step)
        target = self.targets(models, state)
        prediction = self.prediction(models, state)
        values = self.objective(prediction, target, state["mask"])
        loss, total, count = self.reduction(values, state["mask"])
        return dict(
            loss=loss,
            loss_sum=total,
            normalization_count=count,
            num_tokens=state["mask"].sum().detach(),
        )


class Procedure(nn.Module):
    """Role ownership and progressive promotion; student stays on TrainingModule."""

    def __init__(
        self, student, phases, teacher=False, auxiliary=False, promotion_interval=0
    ):
        super().__init__()
        self.phases = nn.ModuleList(phases)
        self.teacher = (
            copy.deepcopy(student).requires_grad_(False).eval() if teacher else None
        )
        self.auxiliary = copy.deepcopy(student) if auxiliary else None
        self.promotion_interval = int(promotion_interval)
        self.register_buffer("last_promotion", torch.tensor(0, dtype=torch.long))
        if not phases:
            raise ValueError("procedure requires at least one phase")
        if not auxiliary and (len(phases) != 1 or phases[0].role != "student"):
            raise ValueError("single-role procedures require one student phase")
        if self.promotion_interval and self.teacher is None:
            raise ValueError("promotion requires a teacher")
        for phase in phases:
            if phase.role not in {"student", "auxiliary"} or (
                phase.role == "auxiliary" and self.auxiliary is None
            ):
                raise ValueError("phase requires a trainable model role")

    def train(self, mode=True):
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    @contextmanager
    def active_role(self, student, role):
        """Freeze inactive parameters without suppressing input derivatives."""
        models = {"student": student, "auxiliary": self.auxiliary}
        flags, modes = [], []
        for name, model in models.items():
            if model is None:
                continue
            modes.append((model, model.training))
            if name != role:
                model.eval()
                for parameter in model.parameters():
                    flags.append((parameter, parameter.requires_grad))
                    parameter.requires_grad_(False)
        try:
            yield
        finally:
            for parameter, flag in flags:
                parameter.requires_grad_(flag)
            for model, mode in modes:
                model.train(mode)

    def forward(self, student, batch, step, phase_index=0):
        if (
            self.promotion_interval
            and step >= int(self.last_promotion) + self.promotion_interval
        ):
            self.teacher.load_state_dict(student.state_dict(), strict=True)
            self.last_promotion.fill_(step)
            for phase in self.phases:
                if hasattr(phase.targets, "step_size"):
                    phase.targets.step_size.mul_(2).clamp_max_(1.0)
        models = dict(student=student, teacher=self.teacher, auxiliary=self.auxiliary)
        return self.phases[phase_index](models, batch, step)

"""Algorithm-side feedback from training predictions to adaptive schedules."""

import torch

from core.parameterizations.plug_in import PlugInParameterization
from core.parameterizations.posterior import PosteriorParameterization


def validate_adaptive_feedback(schedule, predictor):
    if not hasattr(schedule, "observe"):
        return
    if schedule.loss_signal not in {"objective", "cross_entropy"}:
        raise ValueError("adaptive loss_signal must be objective or cross_entropy")
    if schedule.loss_signal == "cross_entropy" and not isinstance(
        predictor, (PosteriorParameterization, PlugInParameterization)
    ):
        raise TypeError(
            "cross-entropy adaptation requires normalized token predictions; "
            "use loss_signal=objective for a score parameterization"
        )


@torch.no_grad()
def observe_training_loss(
    schedule, time, prediction, tokens, per_token, loss_mask, *, step, training
):
    """Send detached per-example feedback, excluding examples without targets.

    The schedule owns fitting; selecting and reducing its observation signal
    belongs to training orchestration. All ranks call observe, even when this
    rank has no valid examples, so the schedule can gather observations safely.
    """
    if not training or not hasattr(schedule, "observe"):
        return
    signal = per_token.detach()
    if schedule.loss_signal == "cross_entropy":
        signal = -prediction.detach().gather(-1, tokens[..., None]).squeeze(-1)
    weights = loss_mask.to(signal)
    counts = weights.sum(-1)
    valid = counts > 0
    masked = torch.where(weights != 0, signal, 0) * weights
    per_example = masked.sum(-1) / counts.clamp_min(1)
    schedule.observe(time.detach()[valid], per_example[valid], step, training=True)

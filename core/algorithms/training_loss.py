"""The shared end of a training loss: token reduction plus adaptive feedback."""

from core.time_sampling.density import (
    objective_weighting,
    training_time_density,
    weight_objective,
)

from .adaptive import observe_training_loss, observes_training_loss


def reduce_token_loss(per_token, loss_mask):
    weights = loss_mask.to(per_token)
    count = weights.sum()
    if count <= 0:
        raise ValueError("loss mask must include at least one target")
    total = (per_token * weights).sum()
    return {
        "loss": total / count,
        "loss_sum": total.detach(),
        "num_tokens": count.detach(),
    }


def finish_training_loss(
    per_token,
    loss_mask,
    *,
    tokens,
    prediction,
    step,
    training,
    time_sampler=None,
    sampler_time=None,
    schedule=None,
    schedule_time=None,
    objective=None,
):
    """Reduce the loss and feed back the same detached per-token signal.

    ``time_sampler`` must be the training time sampler that drew this batch (a
    per-call override, not the algorithm default), and ``sampler_time`` its draw.
    Samplers without ``observe``, or with adaptation off, are unaffected. The
    reduced loss follows the sampler's ``objective_weighting``; observers get the
    unweighted loss and the schedule also the per-row draw density.
    """
    weighted = per_token
    if objective_weighting(time_sampler) != "uncorrected":
        weighted = weight_objective(
            per_token, time_sampler, training_time_density(time_sampler, sampler_time), objective
        )
    result = reduce_token_loss(weighted, loss_mask)
    observe = getattr(time_sampler, "observe", None)
    if training and observe is not None:
        observe(sampler_time, per_token, loss_mask, step)
    if schedule is not None:
        observe_training_loss(
            schedule,
            schedule_time,
            prediction,
            tokens,
            per_token,
            loss_mask,
            step=step,
            training=training,
            density=(
                training_time_density(time_sampler, sampler_time)
                if training and observes_training_loss(schedule)
                else None
            ),
        )
    return result

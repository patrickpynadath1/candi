"""The shared end of a training loss: token reduction plus adaptive feedback."""

from .adaptive import observe_training_loss


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
):
    """Reduce the loss and feed back the same detached per-token signal.

    ``time_sampler`` must be the sampler that drew this batch (a per-call
    override, not the algorithm default), and ``sampler_time`` its draw.
    Samplers without ``observe``, or with adaptation off, are unaffected.
    """
    result = reduce_token_loss(per_token, loss_mask)
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
        )
    return result

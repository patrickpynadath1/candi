"""Optional simplex guidance and confidence commits, copied from legacy."""

import torch


def guided_log_probs(unconditional, conditional, scale):
    """Appendix C.2 classifier-free guidance, in log probability space."""
    if scale < 0:
        raise ValueError("guidance scale must be nonnegative")
    if scale == 0:
        return unconditional.log_softmax(-1)
    if scale == 1:
        return conditional.log_softmax(-1)
    if not torch.isfinite(unconditional).all() or not torch.isfinite(conditional).all():
        raise ValueError(
            "simplex guidance interpolation requires finite log probabilities; disable top-k/top-p filtering when using guidance"
        )
    return ((1 - scale) * unconditional + scale * conditional).log_softmax(-1)


def confidence_commit(state, prediction, s, mask, power):
    if power is not None:
        probabilities = prediction.softmax(-1)
        confidence, greedy = probabilities.max(-1)
        eligible = torch.ones_like(confidence, dtype=torch.bool)
        if mask is not None:
            eligible &= ~mask.squeeze(-1)
        rank = (
            confidence.masked_fill(~eligible, -torch.inf)
            .argsort(-1, descending=True)
            .argsort(-1)
        )
        count = (((1 - s) ** power) * eligible.sum(-1)).floor().long()
        committed = eligible & (rank < count[:, None])
        vertex = torch.nn.functional.one_hot(greedy, state.shape[-1]).to(state).log()
        state = torch.where(committed[..., None], vertex, state)
    return state

"""Masked per-example statistics shared by adaptive training components."""


def masked_example_mean(values, mask=None):
    if mask is None:
        return values.mean(-1)
    weights = mask.to(values)
    return (values * weights).sum(-1) / weights.sum(-1).clamp_min(1)

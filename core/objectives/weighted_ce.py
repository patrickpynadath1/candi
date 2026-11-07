import torch


class WeightedCrossEntropy:
    """MDLM's ELBO-weighted cross entropy.

    Standalone callers pass (keep, derivative); the view-based path passes an
    ObjectiveContext, whose first view's keep-rate coefficients supply them.
    """

    requires_context = True

    def __call__(self, log_probs, targets, keep_rate=None, keep_derivative=None, *, context=None):
        if context is not None:
            keep_rate, keep_derivative = context.coefficients[0]
        if torch.any(keep_rate >= 1):
            raise ValueError("MDLM loss requires a noisy time, not the clean endpoint")
        ce = -log_probs.gather(-1, targets[..., None]).squeeze(-1)
        weight = -keep_derivative / (1 - keep_rate)
        return ce * weight[:, None]

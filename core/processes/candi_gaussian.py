"""CANDI's original frozen-posterior continuous update."""

from .gaussian import GaussianProcess


class CANDIGaussianProcess(GaussianProcess):
    def reverse_posterior(self, state, clean, path_t, path_s, noise):
        del noise
        # Historical CANDI labels the current (larger) value ``sigma_s`` and
        # conditions the score at the destination (smaller) ``sigma_t``.
        # This is intentionally different from a standard VE Euler update.
        current = path_t.sigma.to(state)
        destination = path_s.sigma.to(state)
        delta = current - destination
        score = (state - clean) / destination.square()[:, None, None].clamp_min(1e-12)
        return state - delta[:, None, None] * score

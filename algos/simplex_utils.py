"""Simplex Diffusion Models, arXiv:2609.35553, Algorithms 1--3.

Log-space Gamma draws preserve small-concentration coordinates during thinning.
Zero parameters denote distributions on faces, rather than epsilon smoothing.
"""

import torch
import torch.nn.functional as F


def log_gamma_sample(shape):
    """Marsaglia--Tsang with boosting (Algorithm 3), pathwise differentiable."""
    if (shape < 0).any() or not torch.isfinite(shape).all():
        raise ValueError("Gamma shapes must be finite and nonnegative")
    positive = shape > 0
    safe = torch.where(positive, shape, torch.ones_like(shape))
    boosted = torch.where(safe < 1, safe + 1, safe)
    d = boosted - 1 / 3
    c = (9 * d).rsqrt()
    pending = torch.ones_like(shape, dtype=torch.bool)
    accepted = torch.zeros_like(shape)
    while pending.any():
        z = torch.randn_like(shape)
        u = torch.rand_like(shape).clamp_min(torch.finfo(shape.dtype).tiny)
        base = 1 + c * z
        v = base.clamp_min(torch.finfo(shape.dtype).tiny).pow(3)
        take = (
            pending
            & (base > 0)
            & (
                (u < 1 - 0.0331 * z.pow(4))
                | (u.log() < 0.5 * z.square() + d * (1 - v + v.log()))
            )
        )
        accepted = torch.where(take, d.log() + v.log(), accepted)
        pending = pending & ~take
    boost = torch.rand_like(shape).clamp_min(torch.finfo(shape.dtype).tiny).log()
    value = accepted + torch.where(safe < 1, boost / safe, torch.zeros_like(safe))
    return torch.where(positive, value, torch.full_like(value, -torch.inf))


def log_dirichlet_sample(beta):
    if (beta.sum(-1) <= 0).any():
        raise ValueError("Dirichlet requires positive total concentration")
    value = log_gamma_sample(beta)
    return value - value.logsumexp(-1, keepdim=True)


def log_beta_sample(a, b):
    """Return log W and log(1-W), including point masses at zero/one."""
    a, b = torch.broadcast_tensors(a, b)
    # Coordinates outside a Dirichlet's supporting face are omitted from
    # thinning. Defining their irrelevant multiplier as one avoids 0/0.
    a = torch.where(a + b == 0, torch.ones_like(a), a)
    pair = log_dirichlet_sample(torch.stack((a, b), -1))
    return pair[..., 0], pair[..., 1]


def concentration(t, kind="constant_linear", nu0=0.4, nu1=0.75, pivot=0.8, epsilon=1.0):
    """Section 4.2 variance schedules, plus c=epsilon/(1-alpha)."""
    if kind == "ddim":
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        return epsilon / t
    if kind not in {"constant", "constant_linear"}:
        raise ValueError(f"Unknown concentration schedule: {kind}")
    if not (0 < nu0 < 1 and 0 < nu1 < 1 and 0 <= pivot < 1):
        raise ValueError("nu values must be in (0,1), pivot in [0,1)")
    nu = torch.full_like(t, nu0)
    if kind == "constant_linear":
        nu = nu + ((t - pivot) / (1 - pivot)).clamp(0, 1) * (nu1 - nu0)
    return 1 / nu - 1


def corruption_parameters(clean, t, c, prior):
    return c[..., None, None] * (
        (1 - t[..., None, None]) * clean + t[..., None, None] * prior
    )


def reverse_bridge(log_pt, clean_tokens, s, t, cs, ct, prior, churn):
    """Proposition 3.3 / Algorithm 2, with exact face and churn boundaries.

    s,t,cs,ct have shape [batch]; log_pt is [batch,length,vocabulary].
    The endpoint s=0 is decoded separately by the sampler.
    """
    if not 0 <= churn <= 1:
        raise ValueError("churn must be in [0,1]")
    if not ((s > 0) & (s < t) & (t <= 1)).all():
        raise ValueError("bridge requires 0 < s < t <= 1")
    clean = F.one_hot(clean_tokens, log_pt.shape[-1]).to(log_pt)
    bs = corruption_parameters(clean, s, cs, prior)
    if churn == 1:
        return log_dirichlet_sample(bs)
    bt = corruption_parameters(clean, t, ct, prior)
    rho = (1 - churn) * torch.minimum(torch.ones_like(t), cs * s / (ct * t))
    r = rho[:, None, None]
    innovation = (bs - r * bt).clamp_min(0)
    # A zero-mass innovation can occur when c is constant and rho=1.
    mass = innovation.sum(-1, keepdim=True)
    fallback = torch.zeros_like(innovation)
    fallback[..., 0] = 1
    log_v = log_dirichlet_sample(torch.where(mass > 0, innovation, fallback))
    log_w, log_one_minus_w = log_beta_sample(
        (rho * ct)[:, None].expand(log_pt.shape[:2]),
        (cs - rho * ct).clamp_min(0)[:, None].expand(log_pt.shape[:2]),
    )
    log_b, _ = log_beta_sample(r * bt, (1 - r) * bt)
    thinned = log_pt + log_b
    thinned = thinned - thinned.logsumexp(-1, keepdim=True)
    result = torch.logaddexp(
        log_w[..., None] + thinned, log_one_minus_w[..., None] + log_v
    )
    return result - result.logsumexp(-1, keepdim=True)


def guided_log_probs(unconditional, conditional, scale):
    """Appendix C.2 classifier-free guidance, in log probability space."""
    if scale < 0:
        raise ValueError("guidance scale must be nonnegative")
    if scale == 0:
        return unconditional.log_softmax(-1)
    if scale == 1:
        return conditional.log_softmax(-1)
    return ((1 - scale) * unconditional + scale * conditional).log_softmax(-1)


def dmd_generator_loss(log_pt, auxiliary_prediction, teacher_prediction, a_t):
    """Proposition G.2: differentiate only the reparameterized log state."""
    difference = (auxiliary_prediction - teacher_prediction).detach()
    return a_t[..., None] * (log_pt * difference).sum(-1)

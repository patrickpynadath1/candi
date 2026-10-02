"""Parameter-free architectural choices for Gaussian model inputs.

These functions transform model inputs only; they never change the process path.
"""

import torch


def gaussian_preconditioning(hidden, alpha, sigma):
    """Normalize by the path's total coefficient scale."""
    return hidden * torch.rsqrt(alpha.square() + sigma.square())[:, None, None]


def unit_embedding_preconditioning(hidden, alpha, sigma):
    variance = alpha.square() / hidden.shape[-1] + sigma.square()
    return hidden * variance.rsqrt()[:, None, None]


def no_preconditioning(hidden, alpha, sigma):
    return hidden


def log_one_minus_time(time, sigma, alpha, epsilon):
    probability = time.clamp(0, 1 - max(epsilon, torch.finfo(time.dtype).eps))
    return -torch.log1p(-probability)


def raw_time(time, sigma, alpha, epsilon):
    return time


def log_sigma(time, sigma, alpha, epsilon):
    return sigma.clamp_min(epsilon).log()


def log_effective_noise(time, sigma, alpha, epsilon):
    return log_sigma(time, sigma, alpha, epsilon) - alpha.clamp_min(epsilon).log()


def log_snr(time, sigma, alpha, epsilon):
    return -2 * log_effective_noise(time, sigma, alpha, epsilon)


def negative_log_snr(time, sigma, alpha, epsilon):
    return -log_snr(time, sigma, alpha, epsilon)


INPUT_PRECONDITIONING = {
    "gaussian": gaussian_preconditioning,
    "none": no_preconditioning,
    "unit_embedding": unit_embedding_preconditioning,
}
TIME_CONDITIONING = {
    "log_one_minus_time": log_one_minus_time,
    "time": raw_time,
    "log_sigma": log_sigma,
    "log_effective_noise": log_effective_noise,
    "log_snr": log_snr,
    "negative_log_snr": negative_log_snr,
}


def resolve_choice(choices, name, option):
    try:
        return choices[name]
    except KeyError as error:
        raise ValueError(
            f"{option} must be one of {tuple(choices)}, got {name!r}"
        ) from error

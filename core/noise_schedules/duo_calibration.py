"""DUO Gaussian-to-uniform calibration and its cached derivative convention."""

from dataclasses import dataclass
from pathlib import Path
import math
import pickle
import numpy as np
import torch
from torch import nn
from scipy.special import ndtr, roots_hermitenorm


def build_integral_table(
    vocab_size, gamma_min=-5.0, gamma_max=-1.0, num_points=100000, quadrature_order=512
):
    """Compute the reference Gaussian winning probability and its gamma derivative.

    Gauss-Hermite quadrature evaluates the same integrals as legacy scipy.quad;
    batched computation permits arbitrary task vocabularies without old caches.
    """
    if (
        vocab_size < 2
        or num_points < 2
        or quadrature_order < 2
        or not gamma_min < gamma_max
    ):
        raise ValueError("invalid integral table specification")
    nodes, weights = roots_hermitenorm(quadrature_order)
    weights = weights / np.sqrt(2 * np.pi)
    gammas = np.linspace(gamma_min, gamma_max, num_points)
    values, gradients = [], []
    for batch in np.array_split(gammas, max(1, math.ceil(num_points / 1024))):
        signal = np.exp(-batch / 2)[:, None]
        cdf = ndtr(signal + nodes[None, :]) ** (vocab_size - 1)
        values.append(cdf @ weights)
        gradients.append((-0.5 * signal * nodes[None, :] * cdf) @ weights)
    return dict(
        vocab_size=vocab_size,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        num_points=num_points,
        pt=np.concatenate(values),
        grad_pt=np.concatenate(gradients),
    )


def load_integral_table(path, vocab_size, **build_options):
    if path is None:
        return build_integral_table(vocab_size, **build_options)
    path = Path(path)
    if path.exists():
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                return {key: data[key] for key in data.files}
        # Explicit legacy pickle paths are trusted local artifacts.
        with path.open("rb") as stream:
            return pickle.load(stream)
    if path.suffix != ".npz":
        raise FileNotFoundError(
            f"integral cache not found: {path}; generated caches use .npz"
        )
    data = build_integral_table(vocab_size, **build_options)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement also makes concurrent identical DDP cache builders safe.
    import tempfile

    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".npz", delete=False
    ) as stream:
        temporary = Path(stream.name)
        np.savez(stream, **data)
    temporary.replace(path)
    return data


class CachedIntegral(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gamma, values, gradients, minimum, maximum):
        clipped = gamma.clamp(minimum, maximum)
        indices = torch.round(
            (len(values) - 1) * (clipped - minimum) / (maximum - minimum)
        ).long()
        ctx.save_for_backward(gradients[indices])
        return values[indices]

    @staticmethod
    def backward(ctx, upstream):
        (gradient,) = ctx.saved_tensors
        return upstream * gradient, None, None, None, None


@dataclass(frozen=True)
class DuoNoise:
    gamma: torch.Tensor
    keep: torch.Tensor
    derivative: torch.Tensor
    alpha: torch.Tensor
    sigma: torch.Tensor


class DuoCalibration(nn.Module):
    def __init__(
        self,
        vocab_size,
        cache_path=None,
        gamma_min=-3.5,
        gamma_max=-1.75,
        table_gamma_min=-5.0,
        table_gamma_max=-1.0,
        num_points=100000,
        quadrature_order=512,
        derivative_step=0.001,
    ):
        super().__init__()
        if (
            not all(
                math.isfinite(x)
                for x in (
                    gamma_min,
                    gamma_max,
                    table_gamma_min,
                    table_gamma_max,
                    derivative_step,
                )
            )
            or not gamma_min < gamma_max
            or derivative_step <= 0
        ):
            raise ValueError("invalid DUO gamma bounds or derivative step")
        data = load_integral_table(
            cache_path,
            vocab_size,
            gamma_min=table_gamma_min,
            gamma_max=table_gamma_max,
            num_points=num_points,
            quadrature_order=quadrature_order,
        )
        if int(data.get("vocab_size", -1)) != vocab_size:
            raise ValueError(
                "integral cache vocabulary mismatch or missing vocabulary metadata"
            )
        lo, hi = float(data["gamma_min"]), float(data["gamma_max"])
        values, gradients = np.asarray(data["pt"]), np.asarray(data["grad_pt"])
        if not (lo <= gamma_min < gamma_max <= hi) or not lo < hi:
            raise ValueError(
                "integral cache does not cover the configured gamma interval"
            )
        if (
            values.ndim != 1
            or values.shape != gradients.shape
            or len(values) != int(data["num_points"])
            or len(values) < 2
        ):
            raise ValueError("invalid integral cache array dimensions")
        if (
            not np.isfinite(values).all()
            or not np.isfinite(gradients).all()
            or np.any(values < 1 / vocab_size - 1e-10)
            or np.any(values > 1 + 1e-10)
            or np.any(np.diff(values) > 1e-8)
        ):
            raise ValueError("invalid integral cache probabilities/derivatives")
        self.vocab_size = vocab_size
        self.gamma_min, self.gamma_max = gamma_min, gamma_max
        self.table_min, self.table_max = lo, hi
        self.derivative_step = derivative_step
        self.register_buffer("probability", torch.as_tensor(values.copy()))
        self.register_buffer(
            "probability_derivative", torch.as_tensor(gradients.copy())
        )

    def keep_from_gamma(self, gamma):
        probability = CachedIntegral.apply(
            gamma,
            self.probability,
            self.probability_derivative,
            self.table_min,
            self.table_max,
        )
        return (self.vocab_size * probability - 1) / (self.vocab_size - 1)

    def forward(self, keep, derivative):
        gamma = self.gamma_min + (1 - keep) * (self.gamma_max - self.gamma_min)
        gamma_prime = (self.gamma_max - self.gamma_min) * -derivative
        calibrated = self.keep_from_gamma(gamma)
        calibrated_derivative = (
            gamma_prime
            * (1 / self.derivative_step)
            * (self.keep_from_gamma(gamma + self.derivative_step) - calibrated)
        )
        return DuoNoise(
            gamma,
            calibrated,
            calibrated_derivative,
            torch.sigmoid(-gamma).sqrt(),
            torch.sigmoid(gamma).sqrt(),
        )

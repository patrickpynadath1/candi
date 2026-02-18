import models
import time
import pdb

import torch
import torch.nn.functional as F
from torch.distributions import Normal
import trainer_base
import torch
from omegaconf import DictConfig

def sigma_from_time_vectorized(
    t: torch.Tensor, sigmas: torch.Tensor, errors: torch.Tensor
) -> torch.Tensor:
    """
    Vectorized version of sigma_from_time using linear interpolation.

    Args:
        t: tensor of target Bayes error rates (ε ∈ [0, 1])
        sigmas: 1D tensor of σ values
        errors: 1D tensor of corresponding ε(σ)

    Returns:
        Tensor of interpolated σ values
    """
    t = t.clamp(min=0.0, max=1.0)
    t = t.to(sigmas.device)

    # Normalize ε ∈ [0, 1] to indices in [0, len(errors)-1]
    indices = torch.searchsorted(errors, t, right=True).clamp(1, len(errors) - 1)
    i0 = indices - 1
    i1 = indices

    e0 = errors[i0]
    e1 = errors[i1]
    s0 = sigmas[i0]
    s1 = sigmas[i1]

    interp_t = (t - e0) / (e1 - e0 + 1e-8)
    return s0 + interp_t * (s1 - s0)



def sample_categorical(categorical_probs):
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def expected_rank(d: int, a: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Vectorized expected rank of the true coordinate in a one-hot vector
    corrupted by Gaussian noise.

    Args:
        d: dimension (number of coordinates)
        a: amplitude(s) of the true coordinate, scalar or tensor
        sigma: noise standard deviation(s), scalar or tensor

    Returns:
        Expected rank(s), same broadcasted shape as a and sigma
    """
    # ensure tensors
    a = torch.as_tensor(a, dtype=torch.float32)
    sigma = torch.as_tensor(sigma, dtype=torch.float32)

    # standard normal CDF in torch
    Phi = torch.distributions.Normal(0.0, 1.0).cdf

    # compute p = P(other beats true coordinate)
    p = Phi(-a / (sigma * torch.sqrt(torch.tensor(2.0, dtype=torch.float32))))

    return (d - 1) * p / d


# below is the noise schedule used for embedding diffusion -- follows the VE SDE schedule
def training_sigma_ve(t: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    """
    Map training time t ∈ [0,1] to sigma via log-linear schedule.

    σ(t) = σ_min * (σ_max / σ_min) ** t

    Args:
        t: tensor of shape [...] with values in [0,1]
        sigma_min: minimum noise level
        sigma_max: maximum noise level

    Returns:
        σ(t) with same shape as t
    """
    t = torch.as_tensor(t, dtype=torch.float32)
    log_ratio = torch.log(torch.tensor(sigma_max / sigma_min, dtype=torch.float32).to(t.device))
    return sigma_min * torch.exp(t * log_ratio)


def inference_sigmas(n_steps: int, sigma_min: float, sigma_max: float) -> torch.Tensor:
    """
    Log-linear schedule of sigmas for inference, decreasing from sigma_max to sigma_min.

    σ_k = σ_max * (σ_min / σ_max) ** (k / (n_steps-1))

    Args:
        n_steps: number of inference steps
        sigma_min: minimum noise level
        sigma_max: maximum noise level

    Returns:
        Tensor of shape [n_steps], from σ_max down to σ_min
    """
    ks = torch.linspace(0, 1, n_steps, dtype=torch.float32)
    return sigma_max * (sigma_min / sigma_max) ** ks


def gaussian_to_disc_corruption(K: int, sigma: float) -> float:
    normal = Normal(0., 1.)
    s = torch.linspace(-4*sigma, 1+6*sigma, 1001)
    phi = normal.log_prob((s-1)/sigma).exp() / sigma
    cdf_power = normal.cdf(s/sigma).pow(K-1)
    return 1.0 - torch.trapz(cdf_power * phi, s).item()


def build_error_to_sigma_schedule(
    vocab_size: int,
    sigma_range: tuple = (0.01, 1.0),
    num_points: int = 500,
    device: str = 'cpu'
) -> tuple[torch.Tensor, torch.Tensor]:
    sigmas = torch.linspace(*sigma_range, steps=num_points, device=device)
    errors = torch.tensor(
        [gaussian_to_disc_corruption(vocab_size, s.item()) for s in sigmas], 
        device=device
    )
    return sigmas, errors


def error_to_sigma(
    error_rates: torch.Tensor,
    sigmas: torch.Tensor,
    errors: torch.Tensor
) -> torch.Tensor:
    error_rates = error_rates.clamp(min=0.0, max=1.0)
    
    # Find bracketing indices
    indices = torch.searchsorted(errors, error_rates, right=True).clamp(1, len(errors)-1)
    i0 = indices - 1
    i1 = indices
    
    # Get bracketing values
    e0 = errors[i0]
    e1 = errors[i1]
    s0 = sigmas[i0]
    s1 = sigmas[i1]
    
    # Linear interpolation
    interp_t = (error_rates - e0) / (e1 - e0 + 1e-8)
    return s0 + interp_t * (s1 - s0)


def sigma_to_error(
    sigma_vals: torch.Tensor,
    sigmas: torch.Tensor,
    errors: torch.Tensor
) -> torch.Tensor:
    sigma_vals = sigma_vals.clamp(min=sigmas[0], max=sigmas[-1])
    
    # Find bracketing indices
    indices = torch.searchsorted(sigmas, sigma_vals, right=True).clamp(1, len(sigmas)-1)
    i0 = indices - 1
    i1 = indices
    
    # Get bracketing values
    s0 = sigmas[i0]
    s1 = sigmas[i1]
    e0 = errors[i0]
    e1 = errors[i1]
    
    # Linear interpolation
    interp_t = (sigma_vals - s0) / (s1 - s0 + 1e-8)
    return e0 + interp_t * (e1 - e0)




"""
Semicat (Semi-Categorical Flow Matching) algorithm.

Implements the core semicat flow matching algorithm in the style of
the existing codebase (AR, MDLM, CANDI, etc.), using trainer_base APIs.

Key idea: interpolate between a Gaussian prior x0 ~ N(0, I) and
one-hot encoded clean data x1 via:
    xt = (1 - t) * x0 + t * onehot(x1)
Train a network to predict x1 from (xt, t) with cross-entropy loss.
Sample by integrating the learned vector field from t=0 to t=1.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Literal, cast
from torchdiffeq import odeint

import trainer_base


def _view_for(t: Tensor, other: Tensor) -> Tensor:
    """Reshape t to be broadcastable with other (expand trailing dims)."""
    return t.view(other.size(0), *((1,) * (other.ndim - 1)))


class Semicat(trainer_base.Diffusion):
    """Semi-categorical flow matching.

    Gaussian prior -> linear interpolation -> cross-entropy on x1.
    No discrete noise schedule; uses continuous time t in [0, 1].
    """

    def __init__(self, config, tokenizer):
        vocab_size = len(tokenizer)
        super().__init__(config, tokenizer, vocab_size=vocab_size)
        self.save_hyperparameters()
        self._validate_configuration()

        self.label_smoothing = config.algo.get("label_smoothing", 0.0)
        self.sampling_method = config.algo.get("sampling_method", "flow_map")
        self.temp = config.algo.get("temp", 1.0)

    def _validate_configuration(self):
        return

    # ------------------------------------------------------------------
    # Prior
    # ------------------------------------------------------------------
    def prior_sample(self, *batch_dims) -> Tensor:
        """Sample from isotropic Gaussian prior in R^vocab_size."""
        return torch.randn(
            *batch_dims, self.vocab_size,
            device=self.device, dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------
    @staticmethod
    def _interpolate(x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """Linear interpolation: xt = (1-t)*x0 + t*onehot(x1).

        Uses scatter_add for efficiency (equivalent to the one-hot form).

        Args:
            x0: Gaussian sample, shape (B, L, V).
            x1: Clean token indices, shape (B, L).
            t: Time, shape (B, 1, 1) broadcastable with x0.
        Returns:
            xt of shape (B, L, V).
        """
        assert x0.ndim == x1.ndim + 1
        assert t.ndim == x0.ndim
        xt = x0 * (1.0 - t)
        xt.scatter_add_(-1, x1[..., None], t.expand_as(xt[..., :1]))
        return xt

    # ------------------------------------------------------------------
    # Training loss (VFM cross-entropy)
    # ------------------------------------------------------------------
    def nll(self, x0_tokens, output_tokens, current_accumulation_step=None,
            train_mode=False):
        """Compute the flow-matching cross-entropy loss.

        This replaces the standard nll interface. We ignore output_tokens
        since semicat doesn't use a next-token formulation.

        Args:
            x0_tokens: Clean token ids, shape (B, L).
            output_tokens: Unused (kept for API compat).
        Returns:
            Per-token loss of shape (B, L).
        """
        del output_tokens, current_accumulation_step, train_mode

        B, L = x0_tokens.shape
        x1 = x0_tokens  # clean data

        # Sample Gaussian prior noise
        x0 = torch.randn(
            B, L, self.vocab_size, device=x1.device, dtype=torch.float32
        )

        # Sample time uniformly
        t = torch.rand(B, device=x1.device)
        t_broad = _view_for(t, x0)  # (B, 1, 1)

        # Interpolate
        xt = self._interpolate(x0, x1, t_broad)

        # Predict clean tokens — network receives (xt, t, t) following
        # the original semicat convention where both time args are t
        # for the VFM (non-distillation) case.
        sigma = self._sigma_from_time(t)
        x1_pred = self.backbone(xt, sigma)

        # Cross-entropy loss per token
        # x1_pred: (B, L, V), x1: (B, L)
        loss = F.cross_entropy(
            x1_pred.transpose(-1, 1),  # (B, V, L)
            x1,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )  # (B, L)
        return loss

    def _sigma_from_time(self, t: Tensor) -> Tensor:
        """Convert time t to a sigma-like conditioning value.

        The original semicat passes t directly as the sigma/time
        conditioning to the network. We wrap it here so subclasses
        or different backbones can override.
        """
        return t

    def _process_sigma(self, sigma):
        """API compat — semicat uses raw time as conditioning."""
        return sigma

    # ------------------------------------------------------------------
    # Vector field (for ODE sampling)
    # ------------------------------------------------------------------
    def _vector_field(self, xt: Tensor, t: Tensor) -> Tensor:
        """Compute the learned vector field v(xt, t).

        v(xt, t) = (softmax(net(xt, t, t)) - xt) / (1 - t)
        """
        sigma = self._sigma_from_time(t)
        x1_logits = self.backbone(xt, sigma)
        dr = x1_logits.softmax(dim=-1) - xt
        scale = 1.0 / (1.0 - _view_for(t, xt) + 1e-8)
        return dr * scale

    # ------------------------------------------------------------------
    # Flow map (direct jump)
    # ------------------------------------------------------------------
    def _flow_map_step(
        self, x: Tensor, s: Tensor, t: Tensor
    ) -> Tensor:
        """Compute x_{s,t} via the flow map.

        x_{s,t} = x + (t-s)/(1-s) * (softmax(net(x, s, t)) - x)
        In the VFM (non-distillation) case the network receives (x, s, t)
        but the original code just uses (x, s, t) with the same backbone.
        For our backbone that takes (xt, sigma), we condition on s.
        """
        dt = (t - s) / (1.0 - s + 1e-8)
        sigma_s = self._sigma_from_time(s)
        diff = self.backbone(x, sigma_s).softmax(dim=-1) - x
        return x + _view_for(dt, x) * diff

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_samples(
        self,
        num_samples: int,
        num_steps: int = None,
        sampling_method: str = None,
        prompt_tokens: Tensor = None,
        prompt_mask: Tensor = None,
        **kwargs,
    ) -> Tensor:
        """Generate token sequences.

        Args:
            num_samples: Batch size.
            num_steps: Number of integration steps (default from config).
            sampling_method: "flow_map", "euler", or "dopri5".
                Defaults to self.sampling_method.
            prompt_tokens: Optional prompt token ids for conditional gen.
            prompt_mask: Binary mask (1 = keep prompt token).
        Returns:
            Token ids of shape (B, L).
        """
        if num_steps is None:
            num_steps = self.config.sampling.steps
        if sampling_method is None:
            sampling_method = self.sampling_method

        B = num_samples
        L = self.num_tokens
        x = self.prior_sample(B, L)

        if sampling_method == "flow_map":
            x = self._sample_flow_map(x, B, num_steps)
        elif sampling_method == "euler":
            x = self._sample_euler(x, B, num_steps)
        elif sampling_method == "dopri5":
            x = self._sample_dopri5(x, B, **kwargs)
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")

        return x.argmax(dim=-1)

    def _sample_flow_map(
        self, x: Tensor, batch_size: int, steps: int
    ) -> Tensor:
        """Sample via iterated flow map x_{s,t}."""
        ts = torch.linspace(0.0, 1.0, steps + 1, device=self.device)
        for s, t in zip(ts[:-1], ts[1:]):
            x = self._flow_map_step(
                x,
                s.expand(batch_size),
                t.expand(batch_size),
            )
        return x

    def _sample_euler(
        self, x: Tensor, batch_size: int, steps: int
    ) -> Tensor:
        """Sample via Euler integration of the vector field."""
        ts = torch.linspace(0.0, 1.0, steps + 1, device=self.device)
        for s, t in zip(ts[:-1], ts[1:]):
            s_broad = _view_for(s.expand(batch_size), x)
            v = self._vector_field(x, s_broad.view(-1))
            x = x + (t - s) * v
        return x

    def _sample_dopri5(self, x: Tensor, batch_size: int, **ode_kwargs) -> Tensor:
        """Sample via adaptive dopri5 ODE solver."""
        x = cast(
            Tensor,
            odeint(
                func=self._vector_field,
                y0=x,
                t=torch.tensor([0.0, 1.0], device=self.device),
                method="dopri5",
                **ode_kwargs,
            )[-1],
        )
        return x

    # ------------------------------------------------------------------
    # API compatibility stubs
    # ------------------------------------------------------------------
    def _process_model_input(self, x0, valid_tokens):
        """Semicat doesn't mask or shift tokens."""
        return x0, x0, valid_tokens

    def _process_model_output(self, model_output, xt, sigma):
        """Return log-softmax logits (for eval compatibility)."""
        return (model_output / self.temp).log_softmax(dim=-1)
"""Heat-kernel blurring diffusion over the sequence axis."""

import math

import torch
import torch.nn.functional as F

import utils
from .diffusion import Diffusion
from .registry import algo_registry


def dct_matrix(length, *, device=None, dtype=torch.float32):
    """Return the orthonormal DCT-II synthesis matrix V[n, k]."""
    n = torch.arange(length, device=device, dtype=dtype)[:, None]
    k = torch.arange(length, device=device, dtype=dtype)[None, :]
    matrix = torch.cos(math.pi * (n + 0.5) * k / length)
    scale = torch.full(
        (length,), math.sqrt(2.0 / length), device=device, dtype=dtype
    )
    scale[0] = math.sqrt(1.0 / length)
    return matrix * scale


@algo_registry.register("blurring_diffusion")
class BlurringDiffusion(Diffusion):
    default_sampler = "blurring_ode"
    supported_samplers = frozenset({"blurring_ode"})
    sampling_family = "blurring_diffusion"
    sampling_capabilities = frozenset(
        {"blurring_prior", "blurring_transition"}
    )

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer, vocab_size=len(tokenizer))
        self._validate_configuration()
        self.detach_forward_embedding = bool(
            config.algo.get("detach_forward_embedding", False)
        )
        self.basis = config.algo.get("basis", "dct")
        if self.basis not in {"dct", "isotropic"}:
            raise ValueError("algo.basis must be 'dct' or 'isotropic'")
        self.tau_schedule = config.algo.get("tau_schedule", "linear")
        if self.tau_schedule not in {"linear", "cosine", "exponential"}:
            raise ValueError("unknown blurring tau schedule")
        self.tau_max = float(config.algo.get("tau_max", 8.0))
        terminal_mean = float(config.algo.get("terminal_mean", 1e-3))
        if not 0 < terminal_mean < 1:
            raise ValueError("algo.terminal_mean must lie in (0, 1)")
        self.lambda_floor = -math.log(terminal_mean) / self.tau_max
        self.lambda_scale = float(config.algo.get("lambda_scale", 1.0))
        self.noise_rate = float(config.algo.get("noise_rate", 1.0))
        self.exponential_rate = float(config.algo.get("exponential_rate", 4.0))
        self._dct_cache = {}
        self.save_hyperparameters()

    def _validate_configuration(self):
        if self.config.algo.backbone != "blurring_dit":
            raise ValueError("blurring_diffusion requires backbone=blurring_dit")
        if self.config.algo.parameterization != "mean":
            raise ValueError("blurring_diffusion predicts clean-token marginals")

    def _process_model_input(self, x0, valid_tokens):
        return x0, None, valid_tokens

    def _tau(self, t):
        if self.tau_schedule == "linear":
            return self.tau_max * t, torch.full_like(t, self.tau_max)
        if self.tau_schedule == "cosine":
            angle = math.pi * t / 2
            return (
                self.tau_max * (1 - torch.cos(angle)),
                self.tau_max * math.pi / 2 * torch.sin(angle),
            )
        rate = self.exponential_rate
        denominator = math.expm1(rate)
        return (
            self.tau_max * torch.expm1(rate * t) / denominator,
            self.tau_max * rate * torch.exp(rate * t) / denominator,
        )

    def _rates(self, length, reference):
        if self.basis == "isotropic":
            return torch.full((length,), self.lambda_floor, **reference)
        k = torch.arange(length, **reference)
        laplacian = 2 * (1 - torch.cos(math.pi * k / length))
        return self.lambda_floor + self.lambda_scale * laplacian

    def mode_coefficients(self, t, length):
        tau, tau_prime = self._tau(t)
        reference = {"device": t.device, "dtype": t.dtype}
        rates = self._rates(length, reference)
        a = torch.exp(-tau[:, None] * rates[None, :])
        a_prime = -tau_prime[:, None] * rates[None, :] * a
        remainder = torch.exp(-2 * self.noise_rate * tau)
        sigma = torch.sqrt((1 - remainder).clamp_min(1e-12))
        sigma_prime = (
            self.noise_rate * tau_prime * remainder / sigma.clamp_min(1e-8)
        )
        return a, sigma, a_prime, sigma_prime

    @staticmethod
    def _segments(mask):
        """Return contiguous true spans so padding and conditioning never mix."""
        padded = F.pad(mask.to(torch.int8), (1, 1))
        edges = padded[1:] - padded[:-1]
        starts = (edges == 1).nonzero(as_tuple=False).flatten().tolist()
        ends = (edges == -1).nonzero(as_tuple=False).flatten().tolist()
        return zip(starts, ends)

    def _basis(self, length, reference):
        key = (length, reference.device, reference.dtype)
        if key not in self._dct_cache:
            self._dct_cache[key] = dct_matrix(
                length, device=reference.device, dtype=reference.dtype
            )
        return self._dct_cache[key]

    def _corrupt_segment(self, clean, t):
        length = clean.shape[0]
        basis = self._basis(length, clean)
        modes = basis.t() @ clean
        a, sigma, _, _ = self.mode_coefficients(t[None].to(clean.dtype), length)
        modes = a[0, :, None] * modes + sigma[0] * torch.randn_like(modes)
        return basis @ modes

    def q_xt(self, tokens, t, valid_tokens=None):
        clean = self.backbone.get_embedding(
            tokens, detach=self.detach_forward_embedding
        )
        xt = clean.clone()
        masks = (
            torch.ones_like(tokens, dtype=torch.bool)
            if valid_tokens is None
            else valid_tokens.bool()
        )
        # Sudoku and fixed-length text batches share one generated span.  Keep
        # that common case batched; retain the general per-example path below
        # for padding and irregular conditioning masks.
        if torch.equal(masks, masks[:1].expand_as(masks)):
            spans = list(self._segments(masks[0]))
            if len(spans) == 1:
                start, end = spans[0]
                length = end - start
                basis = self._basis(length, clean)
                modes = torch.matmul(basis.t(), clean[:, start:end])
                a, sigma, _, _ = self.mode_coefficients(t.to(clean.dtype), length)
                noisy_modes = (
                    a[:, :, None] * modes
                    + sigma[:, None, None] * torch.randn_like(modes)
                )
                xt[:, start:end] = torch.matmul(basis, noisy_modes)
                return xt
        for batch in range(tokens.shape[0]):
            for start, end in self._segments(masks[batch]):
                xt[batch, start:end] = self._corrupt_segment(
                    clean[batch, start:end], t[batch]
                )
        return xt

    def forward(self, xt, time):
        return F.log_softmax(self.backbone(xt=xt, time=time), dim=-1)

    def nll(
        self, x0, output_tokens, current_accumulation_step=None,
        train_mode=False, valid_tokens=None,
    ):
        del output_tokens
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        xt = self.q_xt(x0, t, valid_tokens=valid_tokens)
        log_probs = self.forward(xt, t)
        loss = -log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        utils.print_nans(log_probs, "model_output")
        self._record_time_loss(t, loss, valid_tokens, train_mode)
        return loss

    def prior_sample(self, num_samples, length):
        return torch.randn(
            num_samples,
            length,
            self.backbone.input_dim,
            device=self.device,
            dtype=self.backbone.input_embedding.weight.dtype,
        )

    def clean_prompt_embedding(self, tokens):
        return self.backbone.get_embedding(tokens, detach=True)

    @torch.no_grad()
    def sample_step(self, x, t, s, active_mask):
        times = torch.full(
            (x.shape[0],), float(t), device=x.device, dtype=x.dtype
        )
        probabilities = self.forward(x, times).exp()
        estimate = self.backbone.get_embedding(probabilities, detach=True).to(x.dtype)
        updated = x.clone()
        if torch.equal(active_mask, active_mask[:1].expand_as(active_mask)):
            spans = list(self._segments(active_mask[0]))
            if len(spans) == 1:
                start, end = spans[0]
                length = end - start
                basis = self._basis(length, x)
                state_modes = torch.matmul(basis.t(), x[:, start:end])
                clean_modes = torch.matmul(basis.t(), estimate[:, start:end])
                a_t, sigma_t, _, _ = self.mode_coefficients(times, length)
                target_times = torch.full_like(times, float(s))
                a_s, sigma_s, _, _ = self.mode_coefficients(target_times, length)
                residual = (
                    state_modes - a_t[:, :, None] * clean_modes
                ) / sigma_t[:, None, None].clamp_min(1e-6)
                next_modes = (
                    a_s[:, :, None] * clean_modes
                    + sigma_s[:, None, None] * residual
                )
                updated[:, start:end] = torch.matmul(basis, next_modes)
                return updated, probabilities
        for batch in range(x.shape[0]):
            for start, end in self._segments(active_mask[batch]):
                length = end - start
                basis = self._basis(length, x)
                state_modes = basis.t() @ x[batch, start:end]
                clean_modes = basis.t() @ estimate[batch, start:end]
                a_t, sigma_t, _, _ = self.mode_coefficients(
                    times[batch : batch + 1], length
                )
                target_time = torch.full_like(times[batch : batch + 1], float(s))
                a_s, sigma_s, _, _ = self.mode_coefficients(target_time, length)
                # Deterministic Gaussian/DDIM transition.  It follows the same
                # conditional path as the probability-flow field but remains
                # stable as sigma approaches zero at the clean endpoint.
                residual = (
                    state_modes - a_t[0, :, None] * clean_modes
                ) / sigma_t[0].clamp_min(1e-6)
                next_modes = (
                    a_s[0, :, None] * clean_modes + sigma_s[0] * residual
                )
                updated[batch, start:end] = basis @ next_modes
        return updated, probabilities

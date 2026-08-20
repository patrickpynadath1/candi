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
        self.tau_schedule = config.algo.get("tau_schedule", "geometric")
        if self.tau_schedule != "geometric":
            raise ValueError("blurring diffusion requires a geometric tau schedule")
        self.tau_min_attenuation = float(
            config.algo.get("tau_min_attenuation", 0.01)
        )
        self.terminal_nonzero_mode = float(
            config.algo.get("terminal_nonzero_mode", 0.01)
        )
        terminal_mean = float(config.algo.get("terminal_mean", 1e-3))
        if not 0 < terminal_mean < 1:
            raise ValueError("algo.terminal_mean must lie in (0, 1)")
        self.terminal_mean = terminal_mean
        self.lambda_scale = float(config.algo.get("lambda_scale", 1.0))
        self.noise_power = float(config.algo.get("noise_power", 2.0))
        self.crossing_max_decile_fraction = float(
            config.algo.get("crossing_max_decile_fraction", 0.45)
        )
        self._dct_cache = {}
        schedule_length = int(
            config.algo.get(
                "schedule_length", getattr(tokenizer, "seq_len", self.num_tokens)
            )
        )
        self._validate_schedule(schedule_length)
        self.save_hyperparameters()

    def _validate_configuration(self):
        if self.config.algo.backbone != "blurring_dit":
            raise ValueError("blurring_diffusion requires backbone=blurring_dit")
        if self.config.algo.parameterization != "mean":
            raise ValueError("blurring_diffusion predicts clean-token marginals")

    def _process_model_input(self, x0, valid_tokens):
        return x0, None, valid_tokens

    def _schedule_constants(self, length):
        k = torch.arange(length, dtype=torch.float64)
        laplacian = 2 * (1 - torch.cos(math.pi * k / length))
        lambda_max = float(laplacian[-1]) * self.lambda_scale
        lambda_one = float(laplacian[1]) * self.lambda_scale
        tau_min = self.tau_min_attenuation / lambda_max
        tau_end = -math.log(self.terminal_nonzero_mode) / lambda_one
        tau_geometric_max = tau_end + tau_min
        lambda_floor = -math.log(self.terminal_mean) / tau_end
        return tau_min, tau_geometric_max, lambda_floor

    def _tau(self, t, length):
        # Shifted geometric time resolves the spec's endpoint contradiction:
        # it retains log spacing while ensuring tau(0)=0 and therefore a(0)=1.
        tau_min, tau_max, _ = self._schedule_constants(length)
        log_ratio = math.log(tau_max / tau_min)
        unshifted = tau_min * torch.exp(log_ratio * t)
        return unshifted - tau_min, unshifted * log_ratio

    def _rates(self, length, reference):
        _, _, lambda_floor = self._schedule_constants(length)
        if self.basis == "isotropic":
            return torch.full((length,), lambda_floor, **reference)
        k = torch.arange(length, **reference)
        laplacian = 2 * (1 - torch.cos(math.pi * k / length))
        return lambda_floor + self.lambda_scale * laplacian

    def mode_coefficients(self, t, length):
        tau, tau_prime = self._tau(t, length)
        reference = {"device": t.device, "dtype": t.dtype}
        rates = self._rates(length, reference)
        a = torch.exp(-tau[:, None] * rates[None, :])
        a_prime = -tau_prime[:, None] * rates[None, :] * a
        sigma = t.clamp_min(0).pow(self.noise_power)
        sigma_prime = self.noise_power * t.clamp_min(0).pow(
            self.noise_power - 1
        )
        return a, sigma, a_prime, sigma_prime

    @torch.no_grad()
    def _validate_schedule(self, length):
        grid = torch.linspace(0, 1, 10001, dtype=torch.float64)
        a, sigma, _, _ = self.mode_coefficients(grid, length)
        snr = a.square() / sigma[:, None].square().clamp_min(1e-30)
        crossed = snr <= 1
        if not crossed.any(dim=0).all():
            raise ValueError("every positional mode must cross SNR=1 by t=1")
        indices = crossed.to(torch.int8).argmax(dim=0)
        crossing_times = grid[indices]
        if not torch.all(crossing_times[:-1] >= crossing_times[1:]):
            raise ValueError("SNR crossing times must decrease with frequency")
        histogram = torch.histc(crossing_times, bins=10, min=0, max=1).long()
        densest = float(histogram.max()) / length
        if (
            self.basis == "dct"
            and length >= 32
            and densest > self.crossing_max_decile_fraction
        ):
            raise ValueError(
                "blur mode deaths are too bunched: densest decile contains "
                f"{densest:.1%} of modes"
            )
        terminal_mean_snr = float(snr[-1, 0]) * length
        if crossing_times[0] < 0.75:
            raise ValueError("mean positional mode dies too early")
        if terminal_mean_snr >= 0.1:
            raise ValueError(
                f"worst-case terminal mean-mode SNR is {terminal_mean_snr:.4g}"
            )
        # Check the posterior's Markov variance condition on adjacent grid steps.
        ratio = a[1:] / a[:-1].clamp_min(torch.finfo(a.dtype).tiny)
        d2 = sigma[1:, None].square() - ratio.square() * sigma[:-1, None].square()
        if d2.min() < -1e-10:
            raise ValueError("schedule violates reverse-step Markov variance")
        print(
            "Blurring schedule diagnostics: "
            f"L={length}, crossing_deciles={histogram.tolist()}, "
            f"t0={float(crossing_times[0]):.4f}, "
            f"terminal_mean_snr<={terminal_mean_snr:.4g}, "
            f"crossing_times={crossing_times.tolist()}"
        )

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

    def sampling_clean_estimate(self, probabilities):
        """Map p(x0 | xt) to the small embedding space for a reverse step."""
        greedy = bool(self.config.sampling.get("greedy_clean_estimate", False))
        values = probabilities.argmax(-1) if greedy else probabilities
        estimate = self.backbone.get_embedding(values, detach=True)
        if estimate.shape[-1] != self.backbone.input_dim:
            raise RuntimeError("reverse clean estimate must blend the E_in table")
        return estimate

    def reverse_mode_moments(self, state_modes, clean_modes, t, s, length):
        """Exact q(y_s | y_t, x0_hat) moments in the fixed mode basis."""
        a_t, sigma_t, _, _ = self.mode_coefficients(t, length)
        a_s, sigma_s, _, _ = self.mode_coefficients(s, length)
        ratio = a_t / a_s.clamp_min(torch.finfo(a_s.dtype).tiny)
        d2 = sigma_t[:, None].square() - ratio.square() * sigma_s[:, None].square()
        if d2.min() < -1e-6:
            raise RuntimeError("reverse transition has negative variance")
        d2 = d2.clamp_min(0)
        sigma_t2 = sigma_t[:, None].square().clamp_min(1e-20)
        coefficient_state = sigma_s[:, None].square() / sigma_t2 * ratio
        # a_s is the re-blur gate: clean_modes is a sharp t=0 prediction.
        coefficient_prediction = d2 / sigma_t2 * a_s
        if bool(self.config.sampling.get("omit_reblur", False)):
            # Deliberately broken diagnostic requested by the spec.
            coefficient_prediction = d2 / sigma_t2
        mean = (
            coefficient_state[:, :, None] * state_modes
            + coefficient_prediction[:, :, None] * clean_modes
        )
        variance = d2 * sigma_s[:, None].square() / sigma_t2
        return mean, variance

    @torch.no_grad()
    def sample_step(self, x, t, s, active_mask):
        times = torch.full(
            (x.shape[0],), float(t), device=x.device, dtype=x.dtype
        )
        probabilities = self.forward(x, times).exp()
        estimate = self.sampling_clean_estimate(probabilities).to(x.dtype)
        updated = x.clone()
        if torch.equal(active_mask, active_mask[:1].expand_as(active_mask)):
            spans = list(self._segments(active_mask[0]))
            if len(spans) == 1:
                start, end = spans[0]
                length = end - start
                basis = self._basis(length, x)
                state_modes = torch.matmul(basis.t(), x[:, start:end])
                clean_modes = torch.matmul(basis.t(), estimate[:, start:end])
                target_times = torch.full_like(times, float(s))
                mean, variance = self.reverse_mode_moments(
                    state_modes, clean_modes, times, target_times, length
                )
                next_modes = mean
                if float(s) > 0:
                    next_modes = mean + variance.sqrt()[:, :, None] * torch.randn_like(mean)
                updated[:, start:end] = torch.matmul(basis, next_modes)
                return updated, probabilities
        for batch in range(x.shape[0]):
            for start, end in self._segments(active_mask[batch]):
                length = end - start
                basis = self._basis(length, x)
                state_modes = basis.t() @ x[batch, start:end]
                clean_modes = basis.t() @ estimate[batch, start:end]
                target_time = torch.full_like(times[batch : batch + 1], float(s))
                mean, variance = self.reverse_mode_moments(
                    state_modes.unsqueeze(0), clean_modes.unsqueeze(0),
                    times[batch : batch + 1], target_time, length,
                )
                next_modes = mean[0]
                if float(s) > 0:
                    next_modes = next_modes + variance[0].sqrt()[:, None] * torch.randn_like(next_modes)
                updated[batch, start:end] = basis @ next_modes
        return updated, probabilities

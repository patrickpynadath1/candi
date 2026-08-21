"""CANDI with geometric heat-kernel corruption across sequence positions."""

import torch
import torch.nn.functional as F

from .blurring_diffusion import BlurringDiffusion
from .candi import CANDI
from .registry import algo_registry


@algo_registry.register("candi_blurred")
class CANDIBlurred(CANDI):
    """Keep CANDI's discrete process and replace its continuous rank kernel."""

    default_sampler = "candi_entropy_bounded"
    supported_samplers = frozenset({"candi", "candi_entropy_bounded"})

    # Share the corrected, tested heat-kernel mathematics without inheriting
    # BlurringDiffusion's trainer/model adapter.
    _schedule_constants = BlurringDiffusion._schedule_constants
    _tau = BlurringDiffusion._tau
    _rates = BlurringDiffusion._rates
    mode_coefficients = BlurringDiffusion.mode_coefficients
    _segments = staticmethod(BlurringDiffusion._segments)
    _basis = BlurringDiffusion._basis
    reverse_mode_moments = BlurringDiffusion.reverse_mode_moments
    _validate_schedule = BlurringDiffusion._validate_schedule

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        blur = config.algo.blurring
        self.tau_schedule = "geometric"
        self.tau_min_attenuation = float(blur.get("tau_min_attenuation", 0.01))
        self.terminal_nonzero_mode = float(
            blur.get("terminal_nonzero_mode", 0.01)
        )
        self.terminal_mean = float(blur.get("terminal_mean", 0.03))
        self.lambda_scale = float(blur.get("lambda_scale", 1.0))
        self.noise_power = float(blur.get("noise_power", 2.0))
        self.crossing_max_decile_fraction = float(
            blur.get("crossing_max_decile_fraction", 0.45)
        )
        self.basis = blur.get("basis", "dct")
        if self.basis not in {"dct", "isotropic"}:
            raise ValueError("algo.blurring.basis must be dct or isotropic")
        self._dct_cache = {}
        self.generated_length = int(
            blur.get("schedule_length", getattr(tokenizer, "seq_len", self.num_tokens))
        )
        self.generated_start = int(
            blur.get(
                "generated_start", getattr(tokenizer, "prompt_len", 0)
            )
        )
        if self.generated_start + self.generated_length > self.num_tokens:
            raise ValueError("configured generated span exceeds model sequence length")
        self._validate_schedule(self.generated_length)

    def _blur_clean(self, clean, blur_time, valid_tokens=None):
        """Apply q(z_t | x0) within generated spans without padding leakage."""
        result = clean.clone()
        masks = (
            torch.ones(clean.shape[:2], device=clean.device, dtype=torch.bool)
            if valid_tokens is None
            else valid_tokens.bool()
        )
        if torch.equal(masks, masks[:1].expand_as(masks)):
            spans = list(self._segments(masks[0]))
            if len(spans) == 1:
                start, end = spans[0]
                length = end - start
                basis = self._basis(length, clean)
                modes = torch.matmul(basis.t(), clean[:, start:end])
                a, sigma, _, _ = self.mode_coefficients(
                    blur_time.to(clean.dtype), length
                )
                noisy = (
                    a[:, :, None] * modes
                    + sigma[:, None, None] * torch.randn_like(modes)
                )
                result[:, start:end] = torch.matmul(basis, noisy)
                return result
        for batch in range(clean.shape[0]):
            for start, end in self._segments(masks[batch]):
                segment = clean[batch, start:end]
                length = end - start
                basis = self._basis(length, segment)
                modes = basis.t() @ segment
                a, sigma, _, _ = self.mode_coefficients(
                    blur_time[batch : batch + 1].to(clean.dtype), length
                )
                noisy = a[0, :, None] * modes + sigma[0] * torch.randn_like(modes)
                result[batch, start:end] = basis @ noisy
        return result

    def q_xt(self, x, alpha_t, valid_tokens=None, continuous_discrete_noise=None):
        disc_xt = self.discrete_noising(x, alpha_t)
        if valid_tokens is not None:
            disc_xt = torch.where(valid_tokens.bool(), disc_xt, x)
        reveal_mask = (disc_xt == x).float()
        discrete_noise = (1 - alpha_t).reshape(-1)
        blur_time = (
            discrete_noise
            if continuous_discrete_noise is None
            else continuous_discrete_noise.reshape(-1)
        )
        clean = self.backbone.get_embedding(x)
        blurred = self._blur_clean(clean, blur_time, valid_tokens)
        xt = torch.where(reveal_mask.bool().unsqueeze(-1), clean, blurred)
        return {
            "xt": xt,
            "reveal_mask": reveal_mask,
            "discrete_noise": discrete_noise,
            "rank_noise": blur_time,
            # ContDIT uses this scalar for input preconditioning.
            "continuous_noise": blur_time,
        }

    def prior_sample(self, *batch_dims):
        return torch.randn(
            *batch_dims, self.backbone.input_dim,
            dtype=torch.float32, device=self.device,
        )

    def forward(self, **kwargs):
        logits = self.backbone(
            xt=kwargs["xt"], time=kwargs["discrete_noise"]
        )
        log_probs = F.log_softmax(logits / self.temp, dim=-1)
        reveal_mask = kwargs["reveal_mask"].bool()
        if reveal_mask.any():
            table = self.backbone.data_input_table().to(kwargs["xt"])
            revealed_tokens = (kwargs["xt"] @ table.t()).argmax(-1)
            correct = F.one_hot(
                revealed_tokens, num_classes=log_probs.shape[-1]
            ).bool()
            forced = torch.where(
                correct,
                torch.zeros_like(log_probs),
                torch.full_like(log_probs, self.neg_infinity),
            )
            log_probs = torch.where(reveal_mask.unsqueeze(-1), forced, log_probs)
        return log_probs

    def sampling_continuous_schedule(self, timesteps, steps):
        del steps
        path_noise = 1 - self.noise.alpha_t(timesteps)
        if self.factorized_adaptive_schedule:
            return 1 - self.continuous_rank_schedule.alpha_t(timesteps)
        _, blur_time, _, _ = self.corruption_curve(path_noise)
        return blur_time

    def sampling_clean_tokens(self, probabilities):
        """CANDI-blurred commits clean predictions by argmax, never sampling."""
        return probabilities.argmax(dim=-1)

    def sampling_clean_state(self, tokens, reference):
        return self.backbone.get_embedding(tokens, detach=True).to(reference)

    def sampling_state_tokens(self, state):
        table = self.backbone.data_input_table().detach().to(state)
        return (state @ table.t()).argmax(-1)

    def sampling_prompt_state(self, state, prompt_tokens, prompt_mask):
        clean = torch.zeros(state.shape[:2], device=state.device, dtype=torch.bool)
        if prompt_tokens is None or prompt_mask is None:
            return state, clean
        prompt = self.sampling_clean_state(prompt_tokens, state)
        mask = prompt_mask.bool()
        return torch.where(mask.unsqueeze(-1), prompt, state), mask

    @torch.no_grad()
    def _discrete_step(
        self, x_sigma, p_x0, t, dt, prev_clean_mask,
        noise_removal_step=False,
    ):
        s = 0.0 if noise_removal_step else t - dt
        probability = ((t - s) / t.clamp_min(1e-8)).clamp(0, 1)
        newly_clean = (
            torch.rand(prev_clean_mask.shape, device=x_sigma.device) < probability
        ) & ~prev_clean_mask.bool()
        predicted = self.sampling_clean_state(
            self.sampling_clean_tokens(p_x0), x_sigma
        )
        updated = torch.where(newly_clean.unsqueeze(-1), predicted, x_sigma)
        return updated, prev_clean_mask.bool() | newly_clean

    def _sampling_span(self, state):
        start = self.generated_start
        end = start + self.generated_length
        if end > state.shape[1]:
            raise ValueError("sampling state is shorter than the generated span")
        return start, end

    @torch.no_grad()
    def _continuous_step(
        self, x, time_t, time_s, sigma_s, sigma_t,
        clean_mask=None, is_embed=False,
    ):
        del is_embed
        batch = x.shape[0]
        if clean_mask is None:
            clean_mask = torch.zeros(x.shape[:2], device=x.device, dtype=torch.bool)
        discrete_time = torch.full(
            (batch,), float(time_t), device=x.device, dtype=x.dtype
        )
        # The legacy sampler names these arguments from the solver direction:
        # sigma_s is the current (noisier) level and sigma_t the next level.
        current_blur = torch.full(
            (batch,), float(sigma_s), device=x.device, dtype=x.dtype
        )
        next_blur = torch.full(
            (batch,), float(sigma_t), device=x.device, dtype=x.dtype
        )
        log_probs = self.forward(
            xt=x,
            discrete_noise=discrete_time,
            reveal_mask=clean_mask,
            continuous_noise=current_blur,
        )
        probabilities = log_probs.exp()
        clean_tokens = self.sampling_clean_tokens(probabilities)
        clean_state = self.sampling_clean_state(clean_tokens, x)

        start, end = self._sampling_span(x)
        length = end - start
        basis = self._basis(length, x)
        state_modes = torch.matmul(basis.t(), x[:, start:end])
        clean_modes = torch.matmul(basis.t(), clean_state[:, start:end])
        mean, variance = self.reverse_mode_moments(
            state_modes, clean_modes, current_blur, next_blur, length
        )
        next_modes = mean
        if float(sigma_t) > 0:
            next_modes = mean + variance.sqrt()[:, :, None] * torch.randn_like(mean)
        continued = x.clone()
        continued[:, start:end] = torch.matmul(basis, next_modes)
        # Prompt and entropy-accepted coordinates are absorbing clean states.
        continued = torch.where(clean_mask.bool().unsqueeze(-1), x, continued)
        return continued, probabilities

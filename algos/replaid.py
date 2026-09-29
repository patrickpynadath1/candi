"""RePlaid likelihood-based continuous diffusion language model."""

import math

import hydra.utils
import torch
import torch.nn.functional as F

from .registry import algo_registry
from .trainer_base import TrainerBase
from models.ema import ExponentialMovingAverage


class MonotoneGammaSchedule(torch.nn.Module):
    """Differentiable monotone log-SNR schedule with learned endpoints."""

    def __init__(self, gamma_min=-13.3, gamma_max=5.0, hidden_dim=16):
        super().__init__()
        if gamma_max <= gamma_min:
            raise ValueError("gamma_max must exceed gamma_min")
        self.gamma_min = torch.nn.Parameter(torch.tensor(float(gamma_min)))
        delta = math.log(math.expm1(gamma_max - gamma_min))
        self.gamma_delta_raw = torch.nn.Parameter(torch.tensor(delta))
        self.slope_raw = torch.nn.Parameter(torch.zeros(hidden_dim + 1))
        self.centers = torch.nn.Parameter(torch.linspace(-2, 2, hidden_dim))

    def shape_and_derivative(self, t):
        dtype = self.gamma_min.dtype
        t = t.to(dtype)
        base_slope = F.softplus(self.slope_raw[0])
        slopes = F.softplus(self.slope_raw[1:])
        centers = self.centers
        softplus_zero = F.softplus(centers)
        raw = base_slope * t + (
            slopes * (F.softplus(t[:, None] + centers) - softplus_zero)
        ).sum(-1)
        derivative = base_slope + (
            slopes * torch.sigmoid(t[:, None] + centers)
        ).sum(-1)
        one = base_slope + (
            slopes * (F.softplus(1 + centers) - softplus_zero)
        ).sum()
        return raw / one, derivative / one

    def forward(self, t):
        shape, derivative = self.shape_and_derivative(t)
        delta = F.softplus(self.gamma_delta_raw)
        return self.gamma_min + delta * shape, delta * derivative

    def bounds(self):
        return self.gamma_min, self.gamma_min + F.softplus(self.gamma_delta_raw)

    def endpoint_parameters(self):
        return [self.gamma_min, self.gamma_delta_raw]


class PlaidGammaSchedule(torch.nn.Module):
    """Normalized positive-weight tanh MLP, as in the public Plaid model.

    Unlike a positive softplus basis this can represent either curvature.
    Endpoints are independent likelihood parameters; the normalized shape
    receives the Appendix-I second-moment gradient in `_training_gamma`.
    """

    def __init__(self, gamma_min, gamma_max, hidden_dim=1024):
        super().__init__()
        if gamma_max <= gamma_min or hidden_dim < 1:
            raise ValueError("require ordered gamma endpoints and positive width")
        self.gamma_min = torch.nn.Parameter(torch.tensor(float(gamma_min)))
        self.gamma_max = torch.nn.Parameter(torch.tensor(float(gamma_max)))
        self.input_weight = torch.nn.Parameter(torch.randn(hidden_dim))
        self.bias = torch.nn.Parameter(torch.randn(hidden_dim))
        self.output_weight = torch.nn.Parameter(torch.randn(hidden_dim))

    def shape_and_derivative(self, t):
        t = t.to(self.gamma_min)
        positive_in = F.softplus(self.input_weight)
        positive_out = F.softplus(self.output_weight)
        activation = torch.tanh((t[:, None] - 0.5) * positive_in + self.bias)
        lower = torch.tanh(self.bias - 0.5 * positive_in)
        upper = torch.tanh(self.bias + 0.5 * positive_in)
        normalizer = ((upper - lower) * positive_out).sum()
        shape = ((activation - lower) * positive_out).sum(-1) / normalizer
        derivative = ((1 - activation.square()) * positive_in * positive_out).sum(-1) / normalizer
        return shape, derivative

    def bounds(self):
        return self.gamma_min, self.gamma_max

    def endpoint_parameters(self):
        return [self.gamma_min, self.gamma_max]

    def forward(self, t):
        shape, derivative = self.shape_and_derivative(t)
        width = self.gamma_max - self.gamma_min
        return self.gamma_min + width * shape, width * derivative


@algo_registry.register("replaid")
class RePlaid(TrainerBase):
    default_sampler = "replaid_ddpm"
    supported_samplers = frozenset({"replaid_ddpm"})
    sampling_family = "replaid"
    sampling_capabilities = frozenset(
        {
            "replaid_prior",
            "replaid_denoise",
            "replaid_schedule",
            "embedding_projection",
        }
    )

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer, vocab_size=len(tokenizer))
        schedule_type = config.algo.get("schedule_type", "legacy")
        schedules = {"legacy": MonotoneGammaSchedule, "plaid": PlaidGammaSchedule}
        if schedule_type not in schedules:
            raise ValueError(f"unknown RePlaid schedule: {schedule_type}")
        self.gamma_schedule = schedules[schedule_type](
            config.algo.gamma_min,
            config.algo.gamma_max,
            config.algo.schedule_hidden_dim,
        ).double()
        self.register_buffer("recon_std_ema", torch.tensor(1.0))
        self.register_buffer("diff_std_ema", torch.tensor(1.0))
        for component in ("recon", "diff"):
            self.register_buffer(f"{component}_mean_ema", torch.tensor(0.0))
            self.register_buffer(f"{component}_second_ema", torch.tensor(1.0))
        self.self_condition_rate = config.algo.self_condition_rate
        self.split_ema = config.algo.split_ema
        self.output_prior = config.algo.output_prior
        if self.ema is not None:
            self.ema = ExponentialMovingAverage(
                self._get_parameters(), decay=self.config.training.ema
            )
        self._validate_configuration()

    def _validate_configuration(self):
        if self.config.algo.backbone != "replaid_dit":
            raise ValueError("RePlaid requires backbone=replaid_dit")
        if not 0 <= self.config.algo.self_condition_rate <= 1:
            raise ValueError("self_condition_rate must be in [0, 1]")

    def _process_model_input(self, x0, valid_tokens):
        return x0, None, valid_tokens

    def _get_parameters(self):
        if hasattr(self, "gamma_schedule"):
            return list(self.backbone.parameters()) + list(
                self.gamma_schedule.parameters()
            )
        return self.backbone.parameters()

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.backbone.project_embeddings_()

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        if self.config.algo.name == "replaid":
            self.log_dict(self._last_replaid_metrics, on_step=True, on_epoch=False)
        return loss

    def on_before_optimizer_step(self, optimizer):
        del optimizer
        gradient = self.backbone.output_layer.linear.weight.grad
        if gradient is None:
            raise RuntimeError("RePlaid denoiser output head has no gradient; refusing to train a frozen transformer")
        self.log("replaid/head_grad_norm", gradient.detach().norm(), on_step=True, on_epoch=False)

    def sample_t(self, count, device):
        if count == 0:
            return torch.empty(0, device=device)
        offset = torch.rand((), device=device)
        return (offset + torch.arange(count, device=device) / count) % 1

    def _split_size(self, batch_size):
        if batch_size < 2:
            return 0
        ratio = self.recon_std_ema / (self.recon_std_ema + self.diff_std_ema).clamp_min(1e-12)
        return int(torch.round(batch_size * ratio).clamp(1, batch_size - 1).item())

    @torch.no_grad()
    def _update_split_statistics(self, reconstruction, diffusion):
        # Estimate variance across updates, not just inside each sub-batch:
        # a one-example reconstruction sub-batch has zero within-batch std.
        for name, values in (("recon", reconstruction), ("diff", diffusion)):
            mean = getattr(self, f"{name}_mean_ema")
            second = getattr(self, f"{name}_second_ema")
            mean.lerp_(values.mean().to(mean), 1 - self.split_ema)
            second.lerp_(values.square().mean().to(second), 1 - self.split_ema)
            getattr(self, f"{name}_std_ema").copy_(
                (second - mean.square()).clamp_min(1e-12).sqrt()
            )

    def _gamma(self, t):
        return self.gamma_schedule(t.double())

    def _training_gamma(self, t, recon_size):
        """Apply the Appendix-I variance gradient only to schedule-shape params."""
        shape, shape_prime = self.gamma_schedule.shape_and_derivative(t.double())
        if recon_size:
            shape = torch.cat((shape[:recon_size].detach(), shape[recon_size:]))
            shape_prime = torch.cat(
                (shape_prime[:recon_size].detach(), shape_prime[recon_size:])
            )
        variance_weight = [torch.ones_like(shape)]
        if shape.requires_grad:
            shape.register_hook(lambda gradient: gradient * variance_weight[0])
            shape_prime.register_hook(lambda gradient: gradient * variance_weight[0])
        lower, upper = self.gamma_schedule.bounds()
        delta = upper - lower
        gamma = lower + delta * shape
        gamma_prime = delta * shape_prime
        return gamma, gamma_prime, variance_weight

    def q_sample(self, clean_embeddings, gamma, noise=None):
        alpha = torch.sigmoid(-gamma).sqrt().to(clean_embeddings)[:, None, None]
        sigma = torch.sigmoid(gamma).sqrt().to(clean_embeddings)[:, None, None]
        if noise is None:
            noise = torch.randn_like(clean_embeddings)
        return alpha * clean_embeddings + sigma * noise

    def denoise(
        self,
        z_t,
        gamma_t,
        self_condition=None,
        detach_embedding_mask=None,
    ):
        return self.backbone(
            z_t,
            gamma_t,
            self_condition,
            output_prior=self.output_prior,
            detach_embedding_mask=detach_embedding_mask,
        )

    def nll(
        self,
        x0,
        output_tokens,
        current_accumulation_step=None,
        train_mode=False,
        valid_tokens=None,
    ):
        del output_tokens, current_accumulation_step
        batch_size, length = x0.shape
        # Both Monte Carlo terms must be present even for a final singleton
        # batch. Use independent reconstruction/diffusion draws of that example.
        if batch_size == 1:
            mask = None if valid_tokens is None else valid_tokens.expand(2, -1)
            return self.nll(
                x0.expand(2, -1), None, train_mode=train_mode, valid_tokens=mask
            ).mean(0, keepdim=True)
        recon_size = self._split_size(batch_size)
        diffusion_size = batch_size - recon_size
        t = torch.zeros(batch_size, device=x0.device)
        t[recon_size:] = self.sample_t(diffusion_size, x0.device)
        if train_mode:
            gamma, gamma_prime, variance_weight = self._training_gamma(t, recon_size)
        else:
            gamma, gamma_prime = self._gamma(t)
            variance_weight = None
        clean = self.backbone.embed_tokens(x0)
        self_conditioned = torch.zeros(
            batch_size, device=x0.device, dtype=torch.bool
        )
        count = (
            batch_size
            if not train_mode and self.self_condition_rate > 0
            else min(batch_size, math.ceil(self.self_condition_rate * batch_size))
        )
        if count:
            indices = torch.randperm(batch_size, device=x0.device)[:count]
            self_conditioned[indices] = True

        # Plaid Algorithm 1, line 9: self-conditioned rows update only the
        # denoiser. Detach their schedule and input/target embedding paths.
        effective_gamma = torch.where(self_conditioned, gamma.detach(), gamma)
        effective_gamma_prime = torch.where(
            self_conditioned, gamma_prime.detach(), gamma_prime
        )
        effective_clean = torch.where(
            self_conditioned[:, None, None], clean.detach(), clean
        )
        z_t = self.q_sample(effective_clean, effective_gamma)
        context_mask = None
        if self.config.algo.get("clean_context", False) and valid_tokens is not None:
            context_mask = ~valid_tokens.bool().unsqueeze(-1)
            z_t = torch.where(context_mask, effective_clean, z_t)

        self_condition = torch.zeros_like(clean)
        if count:
            with torch.no_grad():
                bootstrap = self.denoise(
                    z_t[indices], effective_gamma[indices].float()
                )
                self_condition[indices] = self.backbone.embed_tokens(
                    bootstrap.exp()
                ).to(self_condition)
            if context_mask is not None:
                self_condition = torch.where(
                    context_mask & self_conditioned[:, None, None],
                    effective_clean.detach(), self_condition,
                )
        log_probs = self.denoise(
            z_t,
            effective_gamma.float(),
            self_condition,
            detach_embedding_mask=self_conditioned,
        )
        predicted = self.backbone.embed_tokens(
            log_probs.exp(), detach_embedding_mask=self_conditioned
        )
        token_ce = -log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1).double()

        # RePlaid precision table: SNR' weighting and final loss aggregation
        # remain float64 even though denoising and reparameterization are fp32.
        snr_prime = -effective_gamma_prime * torch.exp(-effective_gamma)
        squared_error = (predicted - effective_clean).double().square().sum(-1)
        diffusion = -0.5 * snr_prime[:, None] * squared_error
        token_weights = torch.ones_like(token_ce) if valid_tokens is None else valid_tokens.to(token_ce)
        def per_example(value):
            return (value * token_weights).sum(-1) / token_weights.sum(-1).clamp_min(1)

        if variance_weight is not None:
            weights = torch.cat(
                (
                    torch.zeros(recon_size, device=x0.device, dtype=torch.float64),
                    2 * per_example(diffusion)[recon_size:].detach(),
                )
            )
            weights[self_conditioned] = 0
            variance_weight[0] = weights
        gamma1, _ = self._gamma(torch.ones(1, device=x0.device))
        alpha1_sq = torch.sigmoid(-gamma1[0])
        sigma1_sq = 1 - alpha1_sq
        prior = 0.5 * (
            alpha1_sq * clean.double().square().sum(-1)
            + self.config.algo.embedding_dim
            * (sigma1_sq - 1 - torch.log(sigma1_sq))
        )

        if train_mode:
            lower, upper = self.gamma_schedule.bounds()
            self._last_replaid_metrics = {
                "replaid/reconstruction": per_example(token_ce)[:recon_size].mean().detach(),
                "replaid/diffusion": per_example(diffusion)[recon_size:].mean().detach(),
                "replaid/prior": per_example(prior).mean().detach(),
                "replaid/gamma_min": lower.detach(),
                "replaid/gamma_max": upper.detach(),
                "replaid/reconstruction_fraction": recon_size / batch_size,
                "replaid/head_weight_norm": self.backbone.output_layer.linear.weight.detach().norm(),
            }
        losses = prior
        if recon_size:
            losses[:recon_size] += token_ce[:recon_size] * batch_size / recon_size
        if diffusion_size:
            losses[recon_size:] += (
                diffusion[recon_size:] * batch_size / diffusion_size
            )
        if train_mode and recon_size and diffusion_size:
            self._update_split_statistics(
                per_example(token_ce)[:recon_size].detach(),
                per_example(diffusion)[recon_size:].detach(),
            )
        self._record_time_loss(t, losses, valid_tokens, train_mode)
        return losses

    def prior_sample(self, num_samples, length):
        return torch.randn(
            num_samples,
            length,
            self.config.algo.embedding_dim,
            device=self.device,
        )

    def sampling_gamma(self, t):
        return self._gamma(t)[0]

    def sampling_embeddings(self, tokens):
        return self.backbone.embed_tokens(tokens)

    @torch.no_grad()
    def sample_replaid_denoise(self, z_t, t, self_condition=None):
        gamma, _ = self._gamma(t)
        log_probs = self.denoise(z_t, gamma.float(), self_condition)
        return log_probs, self.backbone.embed_tokens(log_probs.exp()), gamma

    def configure_optimizers(self):
        embedding = [self.backbone.token_embedding]
        endpoints = self.gamma_schedule.endpoint_parameters()
        endpoint_ids = {id(parameter) for parameter in endpoints}
        schedule_shape = [parameter for parameter in self.gamma_schedule.parameters()
                          if id(parameter) not in endpoint_ids]
        special_ids = {
            id(parameter)
            for parameter in [*embedding, *schedule_shape, *endpoints]
        }
        backbone = [
            parameter
            for parameter in self.backbone.parameters()
            if id(parameter) not in special_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone, "lr": self.config.optim.lr},
                {
                    "params": [*embedding, *schedule_shape],
                    "lr": self.config.algo.special_lr,
                    "weight_decay": 0.0,
                },
                {
                    "params": endpoints,
                    "lr": self.config.algo.special_lr,
                    "weight_decay": self.config.algo.endpoint_weight_decay,
                },
            ],
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay,
        )
        scheduler = hydra.utils.instantiate(self.config.lr_scheduler, optimizer=optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

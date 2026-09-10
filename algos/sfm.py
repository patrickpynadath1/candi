"""Spherical Flow Matching for language modeling."""

import torch
from omegaconf import OmegaConf, open_dict

from .diffusion import Diffusion
from .noise_schedules import get_noise
from .registry import algo_registry
from .spherical_utils import (
    slerp,
    sphere_normalize,
    spherical_truncation_alpha_max,
)


@algo_registry.register("sfm")
class SFM(Diffusion):
    default_sampler = "sfm"
    supported_samplers = frozenset({"sfm"})
    sampling_family = "sfm"
    sampling_capabilities = frozenset(
        {
            "spherical_prior",
            "spherical_schedule",
            "spherical_embeddings",
            "spherical_logits",
        }
    )

    def __init__(self, config, tokenizer):
        if getattr(config.algo, "auto_truncation", False):
            dimension = int(
                getattr(config.model, "input_embedding_dim", config.model.hidden_size)
            )
            vocab_size = (
                len(tokenizer.added_tokens_decoder)
                if hasattr(config.data, "vocab_file")
                else len(tokenizer)
            )
            alpha_max = spherical_truncation_alpha_max(
                vocab_size,
                dimension,
                float(config.algo.truncation_delta),
            )
            if OmegaConf.is_config(config.noise):
                with open_dict(config.noise):
                    config.noise.alpha_max = alpha_max
            else:
                config.noise.alpha_max = alpha_max
        super().__init__(config, tokenizer)
        self.eps = getattr(config.algo, "eps", 1e-6)
        self.invert_time_convention = getattr(
            config.algo, "invert_time_convention", True
        )
        self.slerp_precision = getattr(config.algo, "slerp_precision", "float32")
        self.renormalize_weights = getattr(config.algo, "renormalize_weights", False)
        self.temp = getattr(config.algo, "temp", 1.0)
        self.noise = get_noise(config)
        self._validate_configuration()

    def _validate_configuration(self):
        if self.invert_time_convention and getattr(
            self.config.noise, "adaptive", False
        ):
            raise ValueError("adaptive schedule requires invert_time_convention=false")
        if self.config.algo.backbone == "sphere_arch" and not self.renormalize_weights:
            raise ValueError("sphere_arch requires renormalize_weights=true")
        if getattr(self.config.algo, "auto_truncation", False):
            expected = spherical_truncation_alpha_max(
                self.vocab_size,
                self.backbone.embed_dim,
                float(self.config.algo.truncation_delta),
            )
            if abs(float(self.config.noise.alpha_max) - expected) > 1e-12:
                raise ValueError("automatic spherical truncation does not match the model")

    def _process_model_output(self, model_output, xt=None, sigma=None):
        del xt, sigma
        return model_output.float().log_softmax(-1)

    def forward(self, xt, sigma):
        return self._process_model_output(self.backbone(xt, self._process_sigma(sigma)))

    def q_xt(self, x, alpha_t, valid_tokens=None):
        clean = self.backbone.get_sphere_embeddings(x)
        noisy = sphere_normalize(torch.randn_like(clean), self.eps)
        amount = alpha_t if self.invert_time_convention else 1 - alpha_t
        if self.slerp_precision == "float64":
            xt = slerp(clean.double(), noisy.double(), amount.double(), self.eps).to(
                clean.dtype
            )
        else:
            xt = slerp(clean, noisy, amount, self.eps)
        if valid_tokens is not None:
            xt = torch.where(valid_tokens.bool().unsqueeze(-1), xt, clean)
        return xt

    def nll_per_token(self, log_x_theta, x0, **kwargs):
        del kwargs
        return -log_x_theta.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

    def nll(
        self,
        input_tokens,
        output_tokens,
        current_accumulation_step=None,
        train_mode=False,
        valid_tokens=None,
    ):
        del output_tokens
        t = self._sample_t(input_tokens.shape[0], current_accumulation_step)
        _, alpha = self.noise(t)
        alpha = alpha.unsqueeze(-1)
        xt = self.q_xt(input_tokens, alpha, valid_tokens=valid_tokens)
        sigma = self._sigma_from_alphat(alpha)
        loss = self.nll_per_token(self.forward(xt, sigma), input_tokens)
        self._record_time_loss(t, loss, valid_tokens, train_mode)
        return loss

    def optimizer_step(self, *args, **kwargs):
        out = super().optimizer_step(*args, **kwargs)
        if self.renormalize_weights:
            self.backbone.renormalize_weights()
        return out

    def prior_sample(self, num_samples, length):
        return sphere_normalize(
            torch.randn(
                num_samples, length, self.backbone.embed_dim, device=self.device
            ),
            self.eps,
        )

    def sampling_alpha(self, t):
        return self.noise(t)[1]

    def sampling_embeddings(self):
        return sphere_normalize(self.backbone.sphere_embed.weight.detach(), self.eps)

    def sampling_logits(self, x, t):
        alpha = self.sampling_alpha(t)
        sigma = self._sigma_from_alphat(alpha).reshape(-1, 1)
        if sigma.shape[0] == 1:
            sigma = sigma.expand(x.shape[0], 1)
        return self.forward(x, sigma) / self.temp

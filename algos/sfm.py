"""Spherical Flow Matching for language modeling."""

import torch

from .diffusion import Diffusion
from .noise_schedules import get_noise
from .registry import algo_registry
from .spherical_utils import slerp, sphere_normalize


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
        super().__init__(config, tokenizer)
        self.eps = getattr(config.algo, "eps", 1e-6)
        self.invert_time_convention = getattr(
            config.algo, "invert_time_convention", True
        )
        self.slerp_precision = getattr(config.algo, "slerp_precision", "float32")
        self.renormalize_weights = getattr(config.algo, "renormalize_weights", False)
        self.noise = get_noise(config)
        self._validate_configuration()

    def _validate_configuration(self):
        if self.invert_time_convention and getattr(
            self.config.noise, "adaptive", False
        ):
            raise ValueError("adaptive schedule requires invert_time_convention=false")
        if self.config.algo.backbone == "sphere_arch" and not self.renormalize_weights:
            raise ValueError("sphere_arch requires renormalize_weights=true")

    def _process_model_output(self, model_output, xt=None, sigma=None):
        del xt, sigma
        return model_output.float().log_softmax(-1)

    def forward(self, xt, sigma):
        return self._process_model_output(self.backbone(xt, self._process_sigma(sigma)))

    def q_xt(self, x, alpha_t):
        clean = self.backbone.get_sphere_embeddings(x)
        noisy = sphere_normalize(torch.randn_like(clean), self.eps)
        amount = alpha_t if self.invert_time_convention else 1 - alpha_t
        if self.slerp_precision == "float64":
            return slerp(clean.double(), noisy.double(), amount.double(), self.eps).to(
                clean.dtype
            )
        return slerp(clean, noisy, amount, self.eps)

    def nll_per_token(self, log_x_theta, x0, **kwargs):
        del kwargs
        return -log_x_theta.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

    def nll(
        self,
        input_tokens,
        output_tokens,
        current_accumulation_step=None,
        train_mode=False,
    ):
        del output_tokens
        t = self._sample_t(input_tokens.shape[0], current_accumulation_step)
        _, alpha = self.noise(t)
        alpha = alpha.unsqueeze(-1)
        xt = self.q_xt(input_tokens, alpha)
        sigma = self._sigma_from_alphat(alpha)
        loss = self.nll_per_token(self.forward(xt, sigma), input_tokens)
        if train_mode:
            self.noise.record_time_loss_pair(
                t, loss.detach().mean(-1), int(self.global_step)
            )
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
        sigma = self._sigma_from_alphat(alpha).expand(x.shape[0], 1)
        return self.forward(x, sigma)

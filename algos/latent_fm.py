"""End-to-end flow matching through a normalized transformer latent space."""

import torch
import torch.nn.functional as F

from .diffusion import Diffusion
from .registry import algo_registry


def _broadcast_time(time, reference):
    return time.view(time.shape[0], *((1,) * (reference.ndim - 1)))


@algo_registry.register("latent_fm")
class LatentFlowMatching(Diffusion):
    default_sampler = "latent_fm"
    supported_samplers = frozenset({"latent_fm"})
    sampling_family = "latent_fm"
    sampling_capabilities = frozenset({"latent_prior", "latent_flow_map"})

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer, vocab_size=len(tokenizer))
        self.label_smoothing = float(config.algo.get("label_smoothing", 0.0))
        self.kl_beta = float(config.algo.get("kl_beta", 0.0))
        if self.kl_beta < 0:
            raise ValueError("algo.kl_beta must be non-negative")
        self.feedback_mode = str(config.algo.get("feedback_mode", "soft"))
        self.feedback_temperature = float(
            config.algo.get("feedback_temperature", 1.0)
        )
        if self.feedback_mode not in {"soft", "sample", "greedy"}:
            raise ValueError("algo.feedback_mode must be soft, sample, or greedy")
        if self.feedback_temperature <= 0:
            raise ValueError("algo.feedback_temperature must be positive")
        self.save_hyperparameters()

    @staticmethod
    def interpolate(noise, clean_latent, clean_time):
        weight = _broadcast_time(clean_time, clean_latent)
        return (1 - weight) * noise + weight * clean_latent

    @staticmethod
    def posterior_kl_values(mean, log_variance):
        """Elementwise KL(q(h|x) || N(0, I))."""
        values = mean.float().square() + log_variance.float().exp()
        values = values - 1 - log_variance.float()
        return 0.5 * values

    @classmethod
    def posterior_kl(cls, mean, log_variance):
        """KL summed over sequence and width, then averaged over the batch."""
        values = cls.posterior_kl_values(mean, log_variance)
        return values.flatten(1).sum(1).mean()

    @classmethod
    def posterior_kl_per_scalar(cls, mean, log_variance):
        return cls.posterior_kl_values(mean, log_variance).mean()

    @staticmethod
    def masked_mean(values, valid_tokens=None):
        if valid_tokens is None:
            return values.mean()
        weights = valid_tokens.to(values)
        return (values * weights).sum() / weights.sum().clamp_min(1)

    def prior_sample(self, *batch_dims):
        return torch.randn(
            *batch_dims,
            self.config.model.hidden_size,
            device=self.device,
            dtype=torch.float32,
        )

    def _process_model_input(self, x0, valid_tokens):
        return x0, x0, valid_tokens

    def nll(
        self,
        input_tokens,
        output_tokens,
        current_accumulation_step=None,
        train_mode=False,
        valid_tokens=None,
    ):
        del output_tokens
        posterior_mean, posterior_log_variance = (
            self.backbone.posterior_from_tokens(input_tokens)
        )
        clean_latent = self.backbone.reparameterize(
            posterior_mean, posterior_log_variance
        )
        kl = self.posterior_kl(posterior_mean, posterior_log_variance)
        kl_per_scalar = self.posterior_kl_per_scalar(
            posterior_mean, posterior_log_variance
        )
        noise = torch.randn_like(clean_latent)
        schedule_time = self._sample_t(
            input_tokens.shape[0], current_accumulation_step
        )
        # The shared schedule runs clean -> noise; latent FM uses 0=noise, 1=clean.
        clean_time = 1 - self._schedule_path_time(schedule_time)
        noisy_latent = self.interpolate(noise, clean_latent, clean_time)
        if valid_tokens is not None:
            noisy_latent = torch.where(
                valid_tokens.bool().unsqueeze(-1), noisy_latent, clean_latent
            )
        logits = self.backbone.decode_latents(noisy_latent, clean_time)
        loss = F.cross_entropy(
            logits.transpose(1, 2),
            input_tokens,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        ce = loss
        loss = ce + self.kl_beta * kl
        if train_mode and getattr(self, "_trainer", None) is not None:
            self.log(
                "latent_fm/ce",
                self.masked_mean(ce, valid_tokens),
                on_step=True,
                on_epoch=False,
            )
            self.log(
                "latent_fm/ce_all_tokens",
                ce.mean(),
                on_step=True,
                on_epoch=False,
            )
            self.log("latent_fm/kl", kl, on_step=True, on_epoch=False)
            self.log(
                "latent_fm/kl_per_scalar",
                kl_per_scalar,
                on_step=True,
                on_epoch=False,
            )
            self.log(
                "latent_fm/weighted_kl",
                self.kl_beta * kl,
                on_step=True,
                on_epoch=False,
            )
        self._record_time_loss(schedule_time, loss, valid_tokens, train_mode)
        return loss

    def sample_latent_flow_map(
        self,
        latent,
        start,
        end,
        *,
        prompt_tokens=None,
        prompt_mask=None,
        feedback_mode=None,
        feedback_temperature=None,
    ):
        logits = self.backbone.decode_latents(latent, start)
        mode = feedback_mode or self.feedback_mode
        temperature = feedback_temperature or self.feedback_temperature
        embeddings = self.backbone.feedback_embeddings(logits, mode, temperature)
        if prompt_tokens is not None and prompt_mask is not None:
            prompt_embeddings = self.backbone.embed_tokens(prompt_tokens)
            embeddings = torch.where(
                prompt_mask.bool().unsqueeze(-1), prompt_embeddings, embeddings
            )
        predicted_clean = self.backbone.encode_embeddings(embeddings)
        coefficient = (end - start) / (1 - start).clamp_min(1e-6)
        latent = latent + _broadcast_time(coefficient, latent) * (
            predicted_clean - latent
        )
        return latent, logits

    # Compatibility methods used by shared evaluation code.
    def _process_sigma(self, sigma):
        return sigma

    def _process_model_output(self, model_output, xt=None, sigma=None):
        del xt, sigma
        return model_output.log_softmax(-1)

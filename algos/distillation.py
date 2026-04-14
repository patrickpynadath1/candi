from .duo import DUO
from .registry import algo_registry

import torch
import torch.nn.functional as F


import copy
import os


@algo_registry.register("distillation")
class Distillation(DUO):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.update_teacher_every = config.algo.update_teacher_every
        self.save_hyperparameters()
        self.teacher = None
        self.teacher_ema = config.algo.teacher_ema
        self.linear_growth_dt = config.algo.linear_growth_dt
        self.linear_growth_min = config.algo.linear_growth_min
        self.linear_growth_max = config.algo.linear_growth_max

    def _validate_configuration(self):
        assert os.path.exists(self.config.algo.integral_cache_path), (
            "The integral cache (Eq. 10 in the paper) for "
            f"the {self.config.data.tokenizer_name_or_path} "
            " tokenizer doesnt exist at "
            f"{self.config.algo.integral_cache_path}. "
            "Please generate it by running the utils.py script, "
            "and ensure the correct path is specified using the "
            "algo.integral_cache_path flag."
        )
        assert self.loss_type in {"kl-fwd", "kl-bwd", "posterior", "kl-posterior"}

    def _maybe_update_teacher_weights(self):
        if self.global_step % self.update_teacher_every != 0:
            return
        if self.teacher_ema:
            self.ema.copy_to(self.teacher.parameters())
        else:
            for better_param, current_param in zip(
                self.backbone.parameters(), self.teacher.parameters()
            ):
                if current_param.requires_grad:
                    current_param.data.copy_(better_param.data)

    @torch.no_grad()
    def _teacher_logits(self, xt, sigma):
        if self.teacher is None:
            self.teacher = copy.deepcopy(self.backbone)
        self._maybe_update_teacher_weights()

        sigma = self._process_sigma(sigma)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            model_output = self.teacher(xt, sigma)
        logits = self._process_model_output(
            model_output=model_output, xt=xt, sigma=sigma
        )
        return logits.detach()

    def _sample_trajectory(self, x0, gamma_t, gamma_s):
        """Computes the noisy sample xt."""
        assert gamma_t.ndim == 1
        assert gamma_s.ndim == 1
        assert x0.ndim == 2
        x0 = F.one_hot(x0, self.vocab_size).to(self.dtype).to(self.device)
        gamma_t = gamma_t.unsqueeze(-1).unsqueeze(-1)
        alpha_t = torch.sigmoid(-gamma_t).sqrt()
        sigma_t = torch.sigmoid(gamma_t).sqrt()

        gamma_s = gamma_s.unsqueeze(-1).unsqueeze(-1)
        alpha_s = torch.sigmoid(-gamma_s).sqrt()
        sigma_s = torch.sigmoid(gamma_s).sqrt()

        epsilon = torch.randn(x0.shape, dtype=torch.float32, device=self.device)
        xt = alpha_t * x0 + sigma_t * epsilon
        xs = alpha_s * x0 + sigma_s * epsilon
        return xt, xs

    def _compute_dt(self):
        if self.linear_growth_dt:
            scale = self.global_step / self.trainer.max_steps
            return self.linear_growth_min + scale * (
                self.linear_growth_max - self.linear_growth_min
            )
        n = self.global_step // self.update_teacher_every
        return 2**n / self.T

    def nll(self, x0, output_tokens, current_accumulation_step=None, train_mode=None):
        del output_tokens, train_mode
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        dt = self._compute_dt()
        t = torch.clip(t + dt, 0, 1)

        gamma_t = self.gamma_min + t * (self.gamma_max - self.gamma_min)
        gamma_s = self.gamma_min + (t - dt) * (self.gamma_max - self.gamma_min)

        usdm_alpha_t = self._gamma_to_alphat(gamma_t)
        usdm_alpha_t = usdm_alpha_t.unsqueeze(-1)
        assert usdm_alpha_t.ndim == 2
        usdm_alpha_s = self._gamma_to_alphat(gamma_s)
        usdm_alpha_s = usdm_alpha_s.unsqueeze(-1)
        assert usdm_alpha_s.ndim == 2

        xt, xs = self._sample_trajectory(x0, gamma_t, gamma_s)
        xt_discrete = xt.argmax(-1)
        xs_discrete = xs.argmax(-1)
        log_x_theta_student = self.forward(
            xt_discrete, sigma=self._sigma_from_alphat(usdm_alpha_t)
        )
        log_x_theta_teacher = self._teacher_logits(
            xs_discrete, sigma=self._sigma_from_alphat(usdm_alpha_s)
        )
        if self.config.training.loss_precision == "float64":
            log_x_theta_student = log_x_theta_student.to(torch.float64)
            log_x_theta_teacher = log_x_theta_teacher.to(torch.float64)
        if self.loss_type == "kl-fwd":
            return (
                log_x_theta_teacher.exp() * (log_x_theta_teacher - log_x_theta_student)
            ).sum(-1)
        elif self.loss_type == "kl-bwd":
            return (
                log_x_theta_student.exp() * (log_x_theta_student - log_x_theta_teacher)
            ).sum(-1)

    def training_step(self, batch, batch_idx):
        self.log(
            name="dt",
            value=self._compute_dt(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return super().training_step(batch, batch_idx)
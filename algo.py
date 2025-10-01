import models
import os
import time
import pdb
import collections
import copy
import pickle

import fsspec
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import trainer_base
import utils
import torch
from omegaconf import DictConfig

torch.serialization.add_safe_globals([DictConfig])


class AR(trainer_base.TrainerBase):
    def __init__(self, config, tokenizer):
        vocab_size = tokenizer.vocab_size
        if not hasattr(tokenizer, "mask_token") or tokenizer.mask_token is None:
            self.mask_index = vocab_size
            vocab_size += 1
        else:
            self.mask_index = tokenizer.mask_token_id
        super().__init__(config, tokenizer, vocab_size=vocab_size)
        self.save_hyperparameters()
        self._validate_configuration()
        self.temp = config.algo.get("temp", 1.0)

    def _validate_configuration(self):
        super()._validate_configuration()
        assert not self.config.algo.time_conditioning
        assert self.config.prior.type == "none"

    def _process_model_input(self, x0, valid_tokens):
        input_tokens = x0[:, :-1]
        output_tokens = x0[:, 1:]
        valid_tokens = valid_tokens[:, 1:]
        return input_tokens, output_tokens, valid_tokens

    def nll(self, input_tokens, output_tokens, current_accumulation_step, *args):
        del current_accumulation_step
        output = self.backbone(input_tokens, None)
        output[:, :, self.mask_index] = self.neg_infinity
        output = output.log_softmax(-1)
        return -output.gather(-1, output_tokens[:, :, None])[:, :, 0]

    def generate_samples(self, num_samples, prompt_mask=None, prompt_tokens=None, **kwargs):
        # precompute token buffer
        num_pred_tokens = self.num_tokens - 1
        x = torch.zeros(
            (num_samples, num_pred_tokens + 1), dtype=torch.long, device=self.device
        )

        if prompt_mask is not None and prompt_tokens is not None:
            x = (prompt_mask * prompt_tokens + (1 - prompt_mask) * x).long()
            start = prompt_mask.sum(1).max().long().item() - 1
        else: 
            start = 0
            x[:, 0] = self.tokenizer.bos_token_id
        # precompute noise
        noise = (
            torch.distributions.Gumbel(0, 1)
            .sample((num_samples, num_pred_tokens, self.vocab_size))
            .to(self.device)
        )
        if self.config.sampling.use_float64:
            noise = noise.to(torch.float64)
        for i in range(start, num_pred_tokens):
            output = self.backbone(x[:, : i + 1], None)
            output[:, :, self.mask_index] = self.neg_infinity
            output = output / self.temp
            output = output.log_softmax(-1)
            y = (output[:, -1, :] + noise[:, i, :]).argmax(-1)
            x[:, i + 1] = y
        return x

    def _process_sigma(self, sigma):
        del sigma
        return None


class MDLM(trainer_base.AbsorbingState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()

    def _validate_configuration(self):
        # ancestral sampling isn't desirable because it's slow
        assert self.sampler == "ancestral_cache" or self.sampler == "posterior"

    def _process_model_output(self, model_output, xt, sigma):
        del sigma
        model_output = model_output / self.temp
        model_output[:, :, self.mask_index] += self.neg_infinity

        # Normalize the model_output such that x.exp() is
        # a probability distribution over vocab_size.
        model_output = model_output - torch.logsumexp(
            model_output, dim=-1, keepdim=True
        )
        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = xt != self.mask_index
        model_output[unmasked_indices] = self.neg_infinity
        model_output[unmasked_indices, xt[unmasked_indices]] = 0
        return model_output

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False):
        del xt
        log_p_theta = torch.gather(
            input=log_x_theta, dim=-1, index=x0[:, :, None]
        ).squeeze(-1)


        return log_p_theta * dalpha_t / (1 - alpha_t)

    def _get_score(self, x, sigma):
        model_output = self.forward(x, sigma)
        # score(x, t) = p_t(y) / p_t(x)
        # => log score(x, t) = log p_t(y) - log p_t(x)

        # case 1: x = masked
        #   (i) y = unmasked
        #     log score(x, t) = log p_\theta(x)|_y + log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))
        #   (ii) y = masked
        #     log score(x, t) = 0

        # case 2: x = unmasked
        #   (i) y != masked, y != x
        #     log score(x_i, t) = - inf
        #   (ii) y = x
        #     log score(x_i, t) = 0
        #   (iii) y = masked token
        #     log score(x_i, t) = - log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))

        log_k = -torch.log(torch.expm1(sigma)).squeeze(-1)
        assert log_k.ndim == 1

        masked_score = model_output + log_k[:, None, None]
        masked_score[:, :, self.mask_index] = 0

        unmasked_score = self.neg_infinity * torch.ones_like(model_output)
        unmasked_score = torch.scatter(
            unmasked_score, -1, x[..., None], torch.zeros_like(unmasked_score[..., :1])
        )
        unmasked_score[:, :, self.mask_index] = -(log_k[:, None] * torch.ones_like(x))

        masked_indices = (x == self.mask_index).to(model_output.dtype)[:, :, None]
        model_output = masked_score * masked_indices + unmasked_score * (
            1 - masked_indices
        )
        return model_output.exp()


class D3PMAbsorb(trainer_base.AbsorbingState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()

    def _validate_configuration(self):
        super()._validate_configuration()
        assert self.noise.type == "log-linear"
        assert self.parameterization == "mean"

    def _process_model_output(self, model_output, xt, sigma):
        del xt
        del sigma
        if self.subs_masking:
            model_output[:, :, self.mask_index] += self.neg_infinity
        return model_output.log_softmax(dim=-1)

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False):
        del dalpha_t
        assert not low_var
        dt = 1 / self.T
        t = 1 - alpha_t  # Only valid for log-linear schedule.
        t = t.clamp(0.0, 1.0 - 1e-4)
        alpha_t = alpha_t + torch.zeros_like(xt)
        alpha_s = t - dt + torch.zeros_like(xt)
        assert alpha_s.shape == xt.shape
        assert alpha_t.shape == xt.shape
        log_x_theta_at_x0 = torch.gather(log_x_theta, -1, x0[:, :, None]).squeeze(-1)
        log_x_theta_at_m = log_x_theta[:, :, self.mask_index]
        x_theta_at_m = log_x_theta_at_m.exp()

        term_1_coef = dt / t
        term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
        term_1_log_dr = log_x_theta_at_x0

        term_2_coef = 1 - dt / t
        term_2_log_nr = term_1_log_nr
        term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)
        L_vb_masked = term_1_coef * (term_1_log_nr - term_1_log_dr) + term_2_coef * (
            term_2_log_nr - term_2_log_dr
        )

        diffusion_loss = self.T * L_vb_masked * (xt == self.mask_index)
        return self._reconstruction_loss(x0) + diffusion_loss


class SEDDAbsorb(trainer_base.AbsorbingState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()

    def _validate_configuration(self):
        super()._validate_configuration()
        assert self.config.sampling.predictor == "analytic"

    def _get_score(self, x, sigma):
        return self.forward(x, sigma).exp()

    def _process_model_output(self, model_output, xt, sigma):
        esigm1_log = (
            torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1)
            .log()
            .to(model_output.dtype)
        )
        # logits shape
        # (batch_size, context_length, vocab_size)
        model_output = (
            model_output
            - esigm1_log[:, None, None]
            - np.log(model_output.shape[-1] - 1)
        )
        # The below scatter operation sets the log score
        # for the input word to 0.
        model_output = torch.scatter(
            model_output, -1, xt[..., None], torch.zeros_like(model_output[..., :1])
        )
        return model_output

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False):
        """Computes the SEDD loss for the Absorbing State Diffusion.

        Args:
          log_x_theta: float torch.Tensor with shape (batch_size,
              context_length, vocab_size),
              log score, output of the denoising network.
          xt: int torch.Tensor with shape (batch_size,
              context_length), input.
          x0: int torch.Tensor with shape (batch_size,
              context_length), input.
          alpha_t: float torch.Tensor with shape (batch_size, 1),
              signal level.
          alpha_t: float torch.Tensor with shape (batch_size, 1),
              signal level.
          dalpha_t: float or float torch.Tensor with shape (batch_size, 1),
              time derivative of signal level.
          low_var: bool, low variance loss during training.

        Returns:
          loss with shape (batch_size, context_length).
        """
        assert not low_var
        masked_indices = xt == self.mask_index
        sigma = self._sigma_from_alphat(alpha_t)
        dsigma = -dalpha_t / alpha_t

        expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
        q_ratio = 1 / expsig_minus_1[masked_indices]

        words_that_were_masked = x0[masked_indices]

        neg_term = q_ratio * torch.gather(
            log_x_theta[masked_indices], -1, words_that_were_masked[..., None]
        ).squeeze(-1)
        score = log_x_theta[masked_indices].exp()
        if self.mask_index == self.vocab_size - 1:
            pos_term = score[:, :-1].sum(dim=-1)
        else:
            pos_term = score[:, : self.mask_index].sum(dim=-1) + score[
                :, self.mask_index + 1 :
            ].sum(dim=-1)
        const = q_ratio * (q_ratio.log() - 1)

        entropy = torch.zeros(*xt.shape, device=xt.device)
        entropy[masked_indices] += pos_term - neg_term + const
        return dsigma * entropy


class DUO_BASE(trainer_base.UniformState):
    def __init__(self, config, tokenizer):
        self.temp = config.algo.get("temp", 1.0)
        super().__init__(config, tokenizer)
        self._validate_configuration()

    def on_save_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = collections.OrderedDict(
            (k, v)
            for k, v in checkpoint["state_dict"].items()
            if not k.startswith("teacher")
        )
        super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = collections.OrderedDict(
            (k, v)
            for k, v in checkpoint["state_dict"].items()
            if not k.startswith("teacher")
        )
        super().on_load_checkpoint(checkpoint)

    def _process_model_output(self, model_output, xt, sigma):
        del xt, sigma
        return (model_output / self.temp).log_softmax(dim=-1)

    def _compute_posterior(self, x, xt, alpha_s, alpha_t):
        """Computes the posterior / approximate posterior.

        Args:
          x: Either clean input `x0` (one-hot),
            or model's predicted `x_theta` of shape (B, L, V).
          xt: The noisy latent (as indices) of shape (B, L).
          alpha_s: Noise level at s of shape (B, [L | 1], 1).
          alpha_t: Noise level at t of shape (B, [L | 1], 1).

        Returns:
          Posterior / approximate posterior of shape (B, L, V).
        """
        if self.config.sampling.use_float64:
            x = x.to(torch.float64)
        if alpha_s.ndim == 2:
            alpha_s = alpha_s.unsqueeze(-1)
        if alpha_t.ndim == 2:
            alpha_t = alpha_t.unsqueeze(-1)
        alpha_ts = alpha_t / alpha_s
        d_alpha = alpha_s - alpha_t
        xt_one_hot = F.one_hot(xt, self.vocab_size).to(self.dtype).to(self.device)
        return (
            alpha_t * self.vocab_size * x * xt_one_hot
            + (alpha_ts - alpha_t) * xt_one_hot
            + d_alpha * x
            + (1 - alpha_ts) * (1 - alpha_s) / self.vocab_size
        ) / (
            alpha_t * self.vocab_size * torch.gather(x, -1, xt[..., None])
            + (1 - alpha_t)
        )

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False):
        assert alpha_t.ndim == 2
        assert x0.ndim == 2
        assert xt.ndim == 2
        assert not torch.is_tensor(dalpha_t) or dalpha_t.ndim == 2
        x_reconst = log_x_theta.exp()
        x_bar_theta = (
            self.vocab_size * alpha_t[:, :, None] * x_reconst + 1 - alpha_t[:, :, None]
        )
        coeff = dalpha_t / (self.vocab_size * alpha_t)
        x_eq_xt = (x0 == xt).float()
        x_neq_xt = 1 - x_eq_xt
        xbar_xt = (1 - alpha_t) + self.vocab_size * alpha_t * x_eq_xt
        xbar_theta_xt = torch.gather(x_bar_theta, -1, xt.unsqueeze(-1)).squeeze(-1)
        xbar_theta_x = torch.gather(x_bar_theta, -1, x0.unsqueeze(-1)).squeeze(-1)
        term1 = self.vocab_size * (1 / xbar_xt - 1 / xbar_theta_xt)

        const = (1 - alpha_t) / (self.vocab_size * alpha_t + 1 - alpha_t)
        term2_coefs = x_eq_xt * const + x_neq_xt
        term2_offset = (
            (self.vocab_size - 1) * const * x_eq_xt - (1 / const) * x_neq_xt
        ) * const.log()
        term2_theta = -term2_coefs * (
            x_bar_theta.log().sum(-1) - self.vocab_size * xbar_theta_xt.log()
        )
        term2_theta = (
            term2_theta
            - self.vocab_size
            * alpha_t
            / (1 - alpha_t)
            * (xbar_theta_x.log() - xbar_theta_xt.log())
            * x_neq_xt
        )
        term2 = term2_theta + term2_offset
        diffusion_loss = coeff * (term1 - term2)
        assert diffusion_loss.ndim == 2
        return diffusion_loss

    def _ancestral_update(self, x, t, dt, p_x0=None, noise_removal_step=False, prompt_mask=None):
        del p_x0
        _, alpha_t = self.noise(t)
        if noise_removal_step:
            alpha_s = torch.ones_like(alpha_t)
        else:
            _, alpha_s = self.noise(t - dt)
        sigma_t = self._sigma_from_alphat(alpha_t)
        assert alpha_t.ndim == 2

        q_xs = self._compute_posterior(
            x=(self.forward(x, sigma_t) / self.temp).exp().double(), xt=x, alpha_s=alpha_s, alpha_t=alpha_t
        )
        if self.p_nucleus < 1:
            q_xs = utils.top_k_top_p_filtering(q_xs.log(), top_p=self.p_nucleus)

        xs = trainer_base.sample_categorical(q_xs)

        # to handle conditional generation for inputs / infilling tasks
        if prompt_mask is not None:
            # expects the original x to have the prompt tokens where prompt_mask = 1
            xs = (xs * (1 - prompt_mask) + x * prompt_mask).long()
        return None, xs


class Integral(torch.autograd.Function):
    """
    torch module calculating UDLM's p_t
    """

    @staticmethod
    def forward(ctx, gamma_t, data):
        gamma_max = data["gamma_max"]
        gamma_min = data["gamma_min"]
        if (gamma_t.max() > gamma_max) or (gamma_t.min() < gamma_min):
            print("max:{} {}".format(gamma_t.max(), gamma_max))
            print("min:{} {}".format(gamma_t.min(), gamma_min))
            gamma_t = torch.clip(gamma_t, gamma_min, gamma_max)
        indices = torch.round(
            (data["num_points"] - 1) * (gamma_t - gamma_min) / (gamma_max - gamma_min)
        ).long()
        grad_pt = data["grad_pt"]
        ctx.grad_pt = grad_pt[indices]

        pt = data["pt"][indices]
        assert pt.shape == gamma_t.shape
        return pt

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.grad_pt * grad_output, None


class DUO(DUO_BASE):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        with fsspec.open(self.config.algo.integral_cache_path, "rb") as f:
            self.integral_cache = pickle.load(f)
        self.integral_cache["pt"] = torch.from_numpy(self.integral_cache["pt"])
        self.integral_cache["grad_pt"] = torch.from_numpy(
            self.integral_cache["grad_pt"]
        )
        self.gamma_min = self.config.algo.gamma_min
        self.gamma_max = self.config.algo.gamma_max
        self.gumbel_tau_log10_start = self.config.algo.gumbel_tau_log10_start
        self.gumbel_tau_log10_end = self.config.algo.gumbel_tau_log10_end
        self.curriculum_start = self.config.algo.curriculum_start
        self.curriculum_end = self.config.algo.curriculum_end
        self.loss_type = self.config.algo.loss_type
        self.temp = config.algo.temp
        self._validate_configuration()

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.integral_cache["pt"] = self.integral_cache["pt"].to(*args, **kwargs)
        self.integral_cache["grad_pt"] = self.integral_cache["grad_pt"].to(
            *args, **kwargs
        )
        return self

    def _compute_gumbel_tau_inverse(self):
        start = self.gumbel_tau_log10_start
        end = self.gumbel_tau_log10_end
        delta = end - start
        if self.global_step < self.curriculum_start:
            tau = start
        elif self.global_step < self.curriculum_end:
            frac = (self.global_step - self.curriculum_start) / (
                self.curriculum_end - self.curriculum_start
            )
            tau = start + frac * delta
        else:
            tau = -10
        return 10 ** (-tau)

    def training_step(self, batch, batch_idx):
        self.log(
            name="gumbel_tau_log10",
            value=1 / self._compute_gumbel_tau_inverse(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return super().training_step(batch, batch_idx)

    def _gamma_to_alphat(self, gamma_t):
        integral = Integral.apply(gamma_t, self.integral_cache)
        return (self.vocab_size * integral - 1) / (self.vocab_size - 1)

    def _prior_loss(self):
        alpha_1 = self._gamma_to_alphat(torch.tensor(self.gamma_max))
        loss = (alpha_1 + (1 - alpha_1) / self.vocab_size) * torch.log(
            (self.vocab_size - 1) * alpha_1 + 1
        ) + (1 - 1 / self.vocab_size) * (1 - alpha_1) * torch.log(1 - alpha_1)
        return loss.item()

    def _q_xt_gaussian(self, x, gamma_t):
        """Computes the noisy sample xt."""
        assert gamma_t.ndim == 1
        assert x.ndim == 3
        gamma_t = gamma_t.unsqueeze(-1).unsqueeze(-1)
        alpha_t = torch.sigmoid(-gamma_t).sqrt()
        sigma_t = torch.sigmoid(gamma_t).sqrt()
        epsilon = torch.randn(x.shape, dtype=torch.float32, device=self.device)
        return alpha_t * x + sigma_t * epsilon

    def nll(self, x0, output_tokens, current_accumulation_step=None, train_mode=False):
        use_true_nll = self.global_step > self.curriculum_end or not train_mode
        if use_true_nll:
            return super().nll(x0, output_tokens, current_accumulation_step)
        del output_tokens
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        gamma_t = self.gamma_min + t * (self.gamma_max - self.gamma_min)
        gamma_t_prime = self.gamma_max - self.gamma_min
        usdm_alpha_t = self._gamma_to_alphat(gamma_t)
        T = 1000
        usdm_dalpha_t = (
            gamma_t_prime * T * (self._gamma_to_alphat(gamma_t + 1 / T) - usdm_alpha_t)
        )
        usdm_alpha_t = usdm_alpha_t.unsqueeze(-1)
        usdm_dalpha_t = usdm_dalpha_t.unsqueeze(-1)
        assert usdm_alpha_t.ndim == 2
        sigma = self._sigma_from_alphat(usdm_alpha_t)

        x0_one_hot = F.one_hot(x0, self.vocab_size)
        xt = self._q_xt_gaussian(x0_one_hot, gamma_t)
        xt = xt * self._compute_gumbel_tau_inverse()
        xt_usdm = xt.argmax(-1)
        log_x_theta = self.forward(xt, sigma=sigma)

        return self.nll_per_token(
            log_x_theta=log_x_theta,
            xt=xt_usdm,
            x0=x0,
            alpha_t=usdm_alpha_t,
            dalpha_t=usdm_dalpha_t,
            low_var=False,
        )


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
    

class CANDI(trainer_base.Diffusion):
    def __init__(self, config, tokenizer):
        self.mask_index = len(tokenizer)
        vocab_size = len(tokenizer) + 1

        if hasattr(config.data, "vocab_file"): 
            vocab_size = len(tokenizer.added_tokens_encoder)
            self.mask_index = vocab_size 
            vocab_size += 1
            self.token_offset = True
        else: 
            self.token_offset = False

        super().__init__(config, tokenizer, vocab_size=vocab_size)
        self._validate_configuration()
        self.save_hyperparameters()
        
        # building error tables
        self.scaling_factor = config.algo.scaling_factor
        self.t_min = config.algo.t_min
        self.t_max = config.algo.t_max
        self.use_actual_elbo = config.algo.use_actual_elbo
        self.use_cfg = config.algo.use_cfg

        self.inference_start_t = config.algo.inference_start_t
        self.inference_end_t = config.algo.inference_end_t
        self.confidence_threshold = config.algo.confidence_threshold

        self.clean_reveal_ratio = config.algo.clean_reveal_ratio
        self.pure_continuous = config.algo.pure_continuous
        # self.use_embedding_cache = config.algo.use_embedding_cache
        # self.continuous_noise, self.discrete_noise = build_sigma_error_table(
        #     vocab_size, 
        #     sigma_range=(.01, 2), 
        #     num=1000, 
        #     # device=self.device, 
        #     scaling_factor=self.scaling_factor
        # )

        self.continuous_noise =torch.linspace(.2, 4.0, 1000)
        self.discrete_noise = expected_rank(d=self.vocab_size-1, a=torch.tensor(1).to(self.device), sigma=self.continuous_noise)

        self.use_percentile_scheduling = config.algo.use_percentile_scheduling

        self.continuous_noise = torch.tensor(self.continuous_noise)
        self.discrete_noise = torch.tensor(self.discrete_noise)
        self.weighted_loss = config.algo.weighted_loss

        # can either be soft (expectation) or hard (sampled from prob dist) 
        self.grad_calc=config.algo.grad_calc
        self.discretize_start_time = config.algo.discretize_start_time

        self.step_size = config.algo.step_size
        self.top_k = config.algo.top_k
        self.only_corrupt_rate = config.algo.only_corrupt_rate

        self.min_percentile = config.algo.min_percentile
        self.max_percentile = config.algo.max_percentile

        self.sigma_min = config.algo.sigma_min
        self.sigma_max = config.algo.sigma_max
        self.temp = config.algo.temp
        self.alpha_beta = config.algo.alpha_beta
        self.use_uniform_sampling = config.algo.use_uniform_sampling
        self.sampler = config.algo.sampler


    def sample_t(self, n, accum_step):
        if self.use_uniform_sampling: 
            return super()._sample_t(n, accum_step) 
        
        else: 
            return torch.distributions.Beta(alpha=self.alpha_beta, beta=self.alpha_beta).sample((n,)).to(self.device)
        return
    

    def get_continuous_from_discrete_noise(self, discrete_noise):
        if self.use_percentile_scheduling:
            target_percentile = discrete_noise * (self.max_percentile - self.min_percentile) + self.min_percentile
            return sigma_from_time_vectorized(target_percentile, self.continuous_noise, self.discrete_noise)
        else: 
            sigmas = training_sigma(discrete_noise, sigma_min=self.sigma_min, sigma_max=self.sigma_max)
            return sigmas

    def to(self, *args, **kwargs): 
        super().to(*args, **kwargs)
        self.continuous_noise = self.continuous_noise.to(*args, **kwargs)
        self.discrete_noise = self.discrete_noise.to(*args, **kwargs)
        return self

    def _validate_configuration(self):
        return

    def _process_model_output(self, xt, model_output, reveal_mask): 
        # if xt is already in token form
        if xt.ndim == 2: 
            xt_tokens = xt
        else: 
            xt_tokens = xt.argmax(dim=-1)
        model_output = model_output / self.temp
        model_output = model_output - torch.logsumexp(
            model_output, dim=-1, keepdim=True
        )
        reveal_mask = reveal_mask.bool()

        model_output[reveal_mask] = self.neg_infinity
        model_output[reveal_mask, xt_tokens[reveal_mask]] = 0
        return model_output

    def nll_per_token(self, 
                     log_x_theta, 
                     x0_tokens,
                     alpha_t, 
                     dalpha_t, 
                     reveal_mask,
                     **kwargs): 
        

        log_p_theta = torch.gather(
            input=log_x_theta, dim=-1, index=x0_tokens[:, :, None]
        ).squeeze(-1)
        sigmas = self.get_continuous_from_discrete_noise(1 - alpha_t).squeeze()

        if self.pure_continuous or not self.weighted_loss:
            nll = -1 * log_p_theta
        else: 
            nll = log_p_theta * dalpha_t / (1 - alpha_t)
        return nll


    def discrete_noising(self, x, alpha_t):
        """Computes the noisy sample xt.

        Args:
          x: int torch.Tensor with shape (batch_size,
              diffusion_model_input_length), input.
          move_chance: float torch.Tensor with shape
            (batch_size, 1).
        """
        move_indices = torch.rand(*x.shape, device=x.device) < 1 - alpha_t
        uniform_tensor = torch.randint(0, self.vocab_size, x.shape, device=x.device)
        xt = torch.where(move_indices, uniform_tensor, x)
        if self.ignore_bos:
            xt[:, 0] = x[:, 0]
        return xt


    # gets noisy sample xt
    def q_xt(self, x, alpha_t):

        disc_xt = self.discrete_noising(x, alpha_t)
        reveal_mask = (disc_xt == x).float()

        discrete_noise = (1 - alpha_t).squeeze()
        continuous_noise = self.get_continuous_from_discrete_noise(discrete_noise) 

        # accounting for the mask token
        onehot = F.one_hot(x, num_classes=self.vocab_size-1).float()

        xt_cont = onehot + continuous_noise[:, None, None].to(x.device) * torch.randn_like(onehot, device=x.device)

        if self.pure_continuous: 
            xt = xt_cont
            reveal_mask = torch.zeros_like(reveal_mask)

            return {
                'xt': xt, 
                'reveal_mask': reveal_mask, 
                'discrete_noise': discrete_noise, 
                'continuous_noise': continuous_noise
            }
        else: 
            xt = onehot * reveal_mask.unsqueeze(-1) + (1 - reveal_mask).unsqueeze(-1) * xt_cont
            return {
                'xt': xt, 
                'reveal_mask': reveal_mask, 
                'discrete_noise': discrete_noise, 
                'continuous_noise': continuous_noise
            }

    def prior_sample(self, *batch_dims): 

        sigma = self.get_continuous_from_discrete_noise(torch.tensor(.999).to(self.device))
        noise = torch.randn(
            *batch_dims, self.vocab_size-1, dtype=torch.float32, device=self.device
        )  * sigma


        return noise
    
    def forward(self, **kwargs): 
        with torch.cuda.amp.autocast(dtype=torch.float32):
            model_output = self.backbone(**kwargs)
        return self._process_model_output(model_output=model_output, xt=kwargs['xt'], reveal_mask=kwargs['reveal_mask'])

    def nll(self, x0, output_tokens, current_accumulation_step=None, train_mode=False):
        del output_tokens
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        assert t.shape[0] == x0.shape[0]
        dalpha_t, alpha_t = self.noise(t)
        alpha_t = alpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2

        noisy_output = self.q_xt(x0, alpha_t)
        log_x_theta = self.forward(**noisy_output)

        utils.print_nans(log_x_theta, "model_output")
        return self.nll_per_token(
            log_x_theta=log_x_theta,
            alpha_t=alpha_t,
            dalpha_t=dalpha_t,
            x0_tokens=x0,
            low_var=train_mode,
            **noisy_output,
        )


    def generate_sample_prompt(self, prompt_tokens, x, prompt_mask): 
        prompt_onehots = torch.nn.functional.one_hot(prompt_tokens, x.size(-1)).float().to(x.device)

        x = prompt_onehots * prompt_mask.unsqueeze(-1) + (1 - prompt_mask).unsqueeze(-1) * x
        return x, prompt_mask
    
    def generate_samples(self, **kwargs): 
        if self.sampler == 'cached': 
            return self.generate_samples_cache(**kwargs)
        elif self.sampler == 'distilled': 
            return self.generate_samples_distilled(**kwargs)
        else:
            return self.generate_samples_nocache(**kwargs)
        

    def generate_samples_distilled(self, num_samples, num_steps=None, eps=1e-5, prompt_tokens=None, prompt_mask=None):
        # pdb.set_trace()
        if num_steps is None:
            num_steps = self.config.sampling.steps

        x = self.prior_sample(num_samples, self.num_tokens)


        clean_mask = torch.zeros((num_samples, self.num_tokens), device=x.device)

        if prompt_tokens is not None and prompt_mask is not None: 
            x, clean_mask = self.generate_sample_prompt(prompt_tokens, x, prompt_mask)

        timesteps = torch.linspace(.999, eps, num_steps + 1, device=self.device)
        if self.use_percentile_scheduling:
            continuous_noise = self.get_continuous_from_discrete_noise(timesteps)
        else: 
            continuous_noise = inference_sigmas(num_steps+1, self.sigma_min, self.sigma_max)
        dt = (1 - eps) / (num_steps)

        self.max_sigma = continuous_noise.max().item()

        self.prev_px0 = None
        for i in range(num_steps): 
            t = timesteps[i]
            s = timesteps[i+1]

            # TODO: this is a bug -- really these should be swapped. I've adjusted the code to make sure it still works
            sigma_s = continuous_noise[i]
            sigma_t = continuous_noise[i+1]

            time_t_vec = torch.ones(x.size(0), device=x.device) * t
            time_s_vec = torch.ones(x.size(0), device=x.device) * s
            sigma_t_vec = torch.ones(x.size(0), device=x.device) * sigma_t

            
            p_x0 = self.forward(xt=x, 
                                discrete_noise=time_t_vec, 
                                reveal_mask=clean_mask,
                                continuous_noise=sigma_t_vec).double()
            x0_hat = sample_categorical(p_x0)
            _, alpha_s = self.noise(time_s_vec)
            
            noisy_outputs = self.q_xt(x0_hat, alpha_t=alpha_s[:, None])
            clean_mask = noisy_outputs['reveal_mask']
            x = noisy_outputs['xt']
            
        return x0_hat

    @torch.no_grad()
    def generate_samples_nocache(self, num_samples, num_steps=None, eps=1e-5, prompt_tokens=None, prompt_mask=None):


        """Generate samples from the model."""
        # pdb.set_trace()
        if num_steps is None:
            num_steps = self.config.sampling.steps

        x = self.prior_sample(num_samples, self.num_tokens)


        clean_mask = torch.zeros((num_samples, self.num_tokens), device=x.device)

        if prompt_tokens is not None and prompt_mask is not None: 
            x, clean_mask = self.generate_sample_prompt(prompt_tokens, x, prompt_mask)

        timesteps = torch.linspace(.999, eps, num_steps + 1, device=self.device)
        if self.use_percentile_scheduling:
            continuous_noise = self.get_continuous_from_discrete_noise(timesteps)
        else: 
            continuous_noise = inference_sigmas(num_steps+1, self.sigma_min, self.sigma_max)
        dt = (1 - eps) / (num_steps)

        self.max_sigma = continuous_noise.max().item()

        self.prev_px0 = None
        for i in range(num_steps): 
            t = timesteps[i]
            s = timesteps[i+1]

            # TODO: this is a bug -- really these should be swapped. I've adjusted the code to make sure it still works
            sigma_s = continuous_noise[i]
            sigma_t = continuous_noise[i+1]

            x_cont, p_x0 = self._continuous_step(x, t, sigma_s=sigma_s, sigma_t=sigma_t, clean_mask=clean_mask, time_s=s)
            
            x, clean_mask = self._discrete_step(x_cont, 
                                                p_x0, t, dt, prev_clean_mask=clean_mask)

        final_tokens = x.argmax(dim=-1)
        if self.token_offset: 
            final_tokens += self.tokenizer.eos_token_id
        return final_tokens
    
    # optimized function with cached embeddings
    @torch.no_grad()
    def generate_samples_cache(self, num_samples, num_steps=None, eps=1e-5, prompt_tokens=None, prompt_mask=None):

        x = self.prior_sample(num_samples, self.num_tokens)
        """Generate samples from the model."""
        # pdb.set_trace()
        if num_steps is None:
            num_steps = self.config.sampling.steps

        embedding_cache = self.backbone.get_embedding(x)



        clean_mask = torch.zeros((num_samples, self.num_tokens), device=x.device, dtype=torch.bool)

        if prompt_tokens is not None and prompt_mask is not None: 
            x, clean_mask = self.generate_sample_prompt(prompt_tokens, x, prompt_mask)


        # for using the beta dist 
        # from scipy.stats import beta
        # u = torch.linspace(1, 0, num_steps + 1).numpy()

        # inverse CDF using SciPy
        # timesteps = torch.Tensor(beta.ppf(u, a=3, b=3)).to(self.device)
        # pdb.set_trace()
        timesteps = torch.linspace(.999, eps, num_steps + 1, device=self.device)
        if self.use_percentile_scheduling:
            continuous_noise = self.get_continuous_from_discrete_noise(timesteps)
        else: 
            continuous_noise = inference_sigmas(num_steps+1, self.sigma_min, self.sigma_max)
        dt = (1 - eps) / (num_steps)

        self.max_sigma = continuous_noise.max().item()

        # manually did a grid search and found in general, 
        # > 64 for cfg is best with small w, 
        # < 64 for cfg is best with larger 2 
        # could be tuned further, just nice defaults
        if num_steps > 64: 
            w = .1
        else: 
            w = .5
        # convert to tokens
        x = x.argmax(dim=-1)
        for i in range(num_steps): 
            t = timesteps[i]
            s = timesteps[i+1]

            # TODO: this is a bug -- really these should be swapped. I've adjusted the code to make sure it still works
            sigma_s = continuous_noise[i]
            sigma_t = continuous_noise[i+1]

            embedding_cache, x0_hat = self._continuous_step_cache(x, t, 
                                                                sigma_s=sigma_s, 
                                                                sigma_t=sigma_t, 
                                                                clean_mask=clean_mask.float(), 
                                                                time_s=s,
                                                                embedding_cache=embedding_cache, 
                                                                w=w, 
                                                                is_first=(i==0))
            is_noise_removal_step = (i == num_steps - 1)
            x, clean_mask = self._discrete_step_optimized(x0_hat, x, t, dt, 
                                                        prev_clean_mask=clean_mask, 
                                                        noise_removal_step=is_noise_removal_step)
        return x
    

    def get_sampling_schedule(self, num_steps):
        return torch.linspace(self.inference_start_t, 
                              self.inference_end_t, 
                              num_steps)

    def _discrete_step_optimized(self, x0_hat, xt, t, dt, prev_clean_mask, noise_removal_step=False): 

        if noise_removal_step: 
            s=0
        else:
            s = t - dt

        # unmasking correspends to 1-alpha(s) / 1-alpha(t)
        # this is just s / t under a log linear schedule
        unmask = torch.rand(prev_clean_mask.shape, device=prev_clean_mask.device) < (t-s)/t
        xt[~prev_clean_mask] = x0_hat[~prev_clean_mask]
        new_clean_mask = prev_clean_mask | unmask
        return xt, new_clean_mask
    

    
    def _discrete_step(self, x_sigma, p_x0, t, dt, prev_clean_mask, noise_removal_step=False): 
        if noise_removal_step: 
            s = 0.0
        else:
            s = t - dt


        t_vec = torch.ones(x_sigma.shape[0], device=x_sigma.device) * t.item()
        s_vec = torch.ones(x_sigma.shape[0], device=x_sigma.device) * s.item()
        mask_probs = torch.ones((x_sigma.shape[0], x_sigma.shape[1], 1), device=x_sigma.device) * (s)
        unmasked_probs = p_x0 * (t_vec - s_vec)[:, None, None]

        q_xs = torch.cat([unmasked_probs, mask_probs], dim=-1)
        _x = sample_categorical(q_xs)
        new_clean_mask = (prev_clean_mask.bool() | (_x != self.mask_index)).float()


        # For tokens that got sampled to real values (not mask), use those
        old_x_tokens=x_sigma.argmax(dim=-1)


        # For tokens that got sampled to mask, keep old tokens but mark as not clean
        sampled_real_tokens = torch.where(_x != self.mask_index, _x, old_x_tokens)
        # Apply copy logic: keep old tokens where prev_clean_mask is True
        updated_tokens = torch.where(prev_clean_mask.bool(), old_x_tokens, sampled_real_tokens)
        updated_x = torch.nn.functional.one_hot(updated_tokens, num_classes=x_sigma.shape[-1]).float().to(x_sigma.device)

        updated_x = updated_x * new_clean_mask.unsqueeze(-1) + (1 - new_clean_mask).unsqueeze(-1) * x_sigma

        return updated_x, new_clean_mask
    

    def _continuous_step(
        self,
        x: torch.Tensor,
        time_t: torch.Tensor,
        time_s: torch.Tensor,
        sigma_s: torch.Tensor,
        sigma_t: torch.Tensor,
        clean_mask: torch.Tensor=None,
    ) -> torch.Tensor:

        dt_cont_vec = torch.ones(x.shape[0], device=x.device) * (sigma_s - sigma_t).item()
        time_t_vec = torch.ones(x.shape[0], device=x.device) * time_t.item()
        sigma_t_vec = torch.ones(x.shape[0], device=x.device) * sigma_t.item()
        if clean_mask is None: 
            clean_mask = torch.zeros(x.shape[:-1], device=x.device)
        cond_denoised = self.forward(xt=x, discrete_noise=time_t_vec, reveal_mask=clean_mask, continuous_noise=sigma_t_vec).double()
        denoised = cond_denoised.exp()
        x0_hat = sample_categorical(denoised)
        x0_hat = torch.nn.functional.one_hot(x0_hat, num_classes=x.shape[-1]).float().to(x.device)
        d = (x - x0_hat) / (sigma_t_vec[:, None, None] ** 2)
        x_cont = x - dt_cont_vec[:, None, None]  * d
        return x_cont, denoised
    

    def _continuous_step_cache(
        self,
        x: torch.Tensor,
        time_t: torch.Tensor,
        time_s: torch.Tensor,
        sigma_s: torch.Tensor,
        sigma_t: torch.Tensor,
        embedding_cache: torch.Tensor,
        clean_mask: torch.Tensor=None,
        is_first=False,
        w=.1,
    ) -> torch.Tensor:
        dt = sigma_s - sigma_t
        time_t_vec = torch.ones(x.shape[0], device=x.device) * time_t.item()
        sigma_t_vec = torch.ones(x.shape[0], device=x.device) * sigma_t.item()
        if clean_mask is None: 
            clean_mask = torch.zeros(x.shape[:-1], device=x.device)
        cond_denoised = self.forward(xt=x, 
                                    discrete_noise=time_t_vec, 
                                    reveal_mask=clean_mask,
                                    continuous_noise=sigma_t_vec,
                                    embedding=embedding_cache).double()
        

        # use classifier free guidance here? 
        if self.use_cfg: 
            if not is_first: 
                uncond_denoised = self.prev_px0
                denoised = (cond_denoised * (1 + w) - uncond_denoised * w).softmax(dim=-1)

            else: 
                self.prev_px0 = cond_denoised.clone()
                denoised = cond_denoised.exp()
        else: 
            denoised = cond_denoised.exp()
        x0_hat = sample_categorical(denoised)
        embedding_hat = self.backbone.get_embedding(x0_hat)
        d = (embedding_cache - embedding_hat) / (sigma_t ** 2)
        new_embedding_cache = embedding_cache - dt* d * self.step_size
        return new_embedding_cache, x0_hat 


def randomly_keep_mask(mask: torch.Tensor, keep_rate: float) -> torch.Tensor:

    if mask.dtype != torch.float:
        mask = mask.float()

    probs = torch.ones_like(mask, device=mask.device) * keep_rate.unsqueeze(-1)
    bernoulli_sample = torch.bernoulli(probs * mask)

    return bernoulli_sample.float() 

def compute_discrete_errors(K, sigma, scaling_factor,**kwargs):
    normal = Normal(0.0, 1.0)
    s = torch.linspace(-4 * sigma, scaling_factor + 6 * sigma, 1001)
    phi = normal.log_prob((s - scaling_factor) / sigma).exp() / sigma
    cdf_power = normal.cdf(s / sigma).pow(K - 1)
    return 1.0 - torch.trapz(cdf_power * phi, s).item()


def build_sigma_error_table(K, sigma_range=(1e-3, 3.0), num=500, device=None, scaling_factor=1.0, **kwargs):
    sigmas = torch.linspace(*sigma_range, steps=num, device=device)
    errors = torch.tensor([compute_discrete_errors(K, s.item(), scaling_factor) for s in sigmas], device=device)
    return sigmas, errors



class CANDI_Distill(CANDI):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.save_hyperparameters()
        self.number_steps = config.algo.number_steps
        self.initial_steps = config.algo.number_steps
        self.distill_steps = config.algo.distill_steps
        self.dt = 1 / self.number_steps
        self.cur_round = 0
        # self.grow_every = config.algo.grow_every
        self.max_distill_steps = config.algo.max_distill_steps
        self.budget = config.algo.budget
        self.adapt_every = config.algo.adapt_every
        growth_factor, _ = compute_growth_schedule(self.adapt_every, self.initial_steps, self.max_distill_steps, self.budget)
        self.growth_factor = growth_factor
        self.teacher = None
        self.last_update_step = 0

        # only load the final weights 
        teacher_weights = torch.load(config.algo.base_path, map_location='cpu')['state_dict']

        # replace backbone. with nothing 
        new_weights = {}
        for k, v in teacher_weights.items():
            if k.startswith('backbone.'):
                new_k = k.replace('backbone.', '')
                new_weights[new_k] = v

        self.backbone.load_state_dict(new_weights)
        self.prepare_teacher_and_student()

    def prepare_teacher_and_student(self): 
        if self.teacher is not None: 
            del self.teacher[0]
        # Step 1: Load empty teacher 
        teacher = models.dit_cont.ContDIT(self.config.model, vocab_size=self.vocab_size, use_mixed_mask=self.config.algo.use_mixed_mask, mixed_coeff=self.config.algo.mixed_coeff)

        teacher.load_state_dict(self.backbone.state_dict())
        teacher.eval()
        teacher.requires_grad_(False)
        self.teacher = [teacher]

    def update_teacher(self): 
        for teacher_param, student_param in zip(self.teacher[0].parameters(), self.backbone.parameters()): 
            teacher_param.data.copy_(student_param.data)
        self.teacher[0].eval()
        self.teacher[0].requires_grad_(False)

    def update_round(self): 
        self.cur_round += 1
        
        # Use gradual growth instead of halving
        self.number_steps = self.number_steps / 2
        self.dt = 1 / self.number_steps
        
        print("Updated to round {}, dt {:.6f}, number_steps {:.1f}".format(
            self.cur_round, self.dt, self.number_steps))


    def _teacher_forward(self, **kwargs):
        with torch.cuda.amp.autocast(dtype=torch.float32):
            model_output = self.teacher[0](**kwargs)
        if kwargs.get('ret_embedding', False): 
            return self._process_model_output(model_output=model_output[0], xt=kwargs['xt'], reveal_mask=kwargs['reveal_mask']), model_output[1]
        return self._process_model_output(model_output=model_output, xt=kwargs['xt'], reveal_mask=kwargs['reveal_mask'])
    
    def forward(self, **kwargs):
        with torch.cuda.amp.autocast(dtype=torch.float32):
            model_output = self.backbone(**kwargs)
        if kwargs.get('ret_embedding', False): 
            return self._process_model_output(model_output=model_output[0], xt=kwargs['xt'], reveal_mask=kwargs['reveal_mask']), model_output[1]
        return self._process_model_output(model_output=model_output, xt=kwargs['xt'], reveal_mask=kwargs['reveal_mask'])


    def _continuous_step_teacher(
        self,
        x: torch.Tensor,
        time_t: torch.Tensor,
        time_s: torch.Tensor,
        sigma_s: torch.Tensor,
        sigma_t: torch.Tensor,
        embedding_cache: torch.Tensor,
        clean_mask: torch.Tensor=None,
    ) -> torch.Tensor:
        dt = sigma_s - sigma_t
        time_t_vec = time_t 
        sigma_t_vec = sigma_t 
        if clean_mask is None: 
            clean_mask = torch.zeros(x.shape[:-1], device=x.device)
        denoised = self._teacher_forward(xt=x, 
                                    discrete_noise=time_t_vec, 
                                    reveal_mask=clean_mask,
                                    continuous_noise=sigma_t_vec,
                                    embedding=embedding_cache).double()
        x0_hat = sample_categorical(denoised)
        embedding_hat = self.backbone.get_embedding(x0_hat)
        d = (embedding_cache - embedding_hat) / (sigma_t[:, None, None] ** 2)
        new_embedding_cache = embedding_cache - dt[:, None, None] * d * self.step_size
        return new_embedding_cache, x0_hat, denoised
    
    def _continuous_step_teacher_embedding(
        self,
        x: torch.Tensor,
        time_t: torch.Tensor,
        time_s: torch.Tensor,
        sigma_s: torch.Tensor,
        sigma_t: torch.Tensor,
        embedding_cache: torch.Tensor,
        clean_mask: torch.Tensor=None,
    ) -> torch.Tensor:
        dt = sigma_s - sigma_t
        time_t_vec = time_t 
        sigma_t_vec = sigma_t 
        if clean_mask is None: 
            clean_mask = torch.zeros(x.shape[:-1], device=x.device)
        denoised, embedding_output = self._teacher_forward(xt=x, 
                                    discrete_noise=time_t_vec, 
                                    reveal_mask=clean_mask,
                                    continuous_noise=sigma_t_vec,
                                    embedding=embedding_cache, ret_embedding=True)
        embedding_hat = self.backbone.get_embedding(denoised.double().exp())
        d = (embedding_cache - embedding_hat) / (sigma_t[:, None, None] ** 2)
        new_embedding_cache = embedding_cache - dt[:, None, None] * d * self.step_size
        return new_embedding_cache, denoised, embedding_output


    def _discrete_step_teacher(self, x0_hat, xt, t, dt, prev_clean_mask, noise_removal_step=False): 

        s = (t - dt).clamp(min=0.0)

        unmask = torch.rand(prev_clean_mask.shape, device=prev_clean_mask.device) < ((t-s)/t)[:, None]
        xt[~prev_clean_mask] = x0_hat[~prev_clean_mask]
        new_clean_mask = prev_clean_mask | unmask
        return xt, new_clean_mask

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.teacher[0] = self.teacher[0].to(*args, **kwargs)
        return self


    @torch.no_grad()
    def sample_teacher_trajectories(self, x0, noisy_output, t_start, distill_steps=2, eps=1e-5):
        """
        Generate teacher trajectories for CANDI's hybrid continuous-discrete method.
        
        Args:
            x0: Clean tokens [batch_size, seq_len] 
            t_start: Starting timestep [batch_size, 1]
            distill_steps: Number of teacher denoising steps
            eps: Minimum timestep
            
        Returns:
            teacher_predictions: Log probabilities from teacher trajectory [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = x0.shape
        device = x0.device
        
        # Create timestep schedule for each batch element
        dt = (1 - eps) / self.config.sampling.steps
        
        # Create time schedule spanning distill_steps for each batch element
        # t_start: [batch_size, 1] -> [1, batch_size, 1] 
        # space: [distill_steps+1, 1, 1]
        space = torch.linspace(1, 0, distill_steps + 1, device=device)[:, None, None]
        t_start_expanded = t_start[None, :, :]  # [1, batch_size, 1]
        t_end = torch.clamp(t_start_expanded - dt * distill_steps, min=eps)
        
        # Interpolate: [distill_steps+1, batch_size, 1]
        timesteps = t_start_expanded * space + (1 - space) * t_end
        
        # Initialize with corrupted input using t_start
        dalpha_t, alpha_t = self.noise(t_start)
        alpha_t = alpha_t.unsqueeze(-1)
        
        x = noisy_output['xt']  # Current state (continuous representation)
        clean_mask = noisy_output['reveal_mask'].bool()  # Which tokens are clean
        
        # Storage for teacher predictions
        teacher_predictions = torch.zeros(
            (batch_size, seq_len, x.size(-1)), device=device
        )
        tokens_denoised = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        embedding_cache = self.backbone.get_embedding(x)


        x = x.argmax(dim=-1)
        # Run teacher trajectory
        for i in range(distill_steps):
            t = timesteps[i].squeeze(-1)  # [batch_size]
            s = timesteps[i+1].squeeze(-1)  # [batch_size]
            
            # Get continuous noise for each batch element's timestep
            continuous_noise_t = self.get_continuous_from_discrete_noise(t)
            continuous_noise_s = self.get_continuous_from_discrete_noise(s)
            
            # Continuous step - get teacher's predictions
            embedding_cache, x0_hat, p_x0 = self._continuous_step_teacher(x, 
                                                                 sigma_s = continuous_noise_t, 
                                                                 sigma_t=continuous_noise_s,
                                                                 time_s=s,
                                                                 time_t=t,
                                                                 embedding_cache=embedding_cache,
                                                                 clean_mask=clean_mask) 
            
            # Discrete step - sample updates  
            new_x, new_clean_mask = self._discrete_step_teacher(x0_hat, 
                                                                x, t, dt, 
                                                                prev_clean_mask=clean_mask)
            # Track which tokens got denoised this step
            newly_denoised = (~tokens_denoised) & (new_clean_mask.bool() & ~clean_mask.bool())


            
            # Store teacher predictions for newly denoised tokens
            if newly_denoised.any():
                teacher_predictions[newly_denoised] = p_x0[newly_denoised].float()
            tokens_denoised = tokens_denoised | newly_denoised
            
            # Update state
            x = new_x
            clean_mask = new_clean_mask
        
        # For any remaining masked tokens, use final predictions
        still_masked = ~tokens_denoised
        if still_masked.any():
            teacher_predictions[still_masked] = p_x0[still_masked].float()
        return teacher_predictions
    
    @torch.no_grad()
    def sample_teacher_trajectories_embeddings(self, x0, noisy_output, t_start, distill_steps=2, eps=1e-5):
        """
        Generate teacher trajectories for CANDI's hybrid continuous-discrete method.
        
        Args:
            x0: Clean tokens [batch_size, seq_len] 
            t_start: Starting timestep [batch_size, 1]
            distill_steps: Number of teacher denoising steps
            eps: Minimum timestep
            
        Returns:
            teacher_predictions: Log probabilities from teacher trajectory [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = x0.shape
        device = x0.device
        
        # Create timestep schedule for each batch element
        dt = (1 - eps) / self.config.sampling.steps
        
        # Create time schedule spanning distill_steps for each batch element
        # t_start: [batch_size, 1] -> [1, batch_size, 1] 
        # space: [distill_steps+1, 1, 1]
        space = torch.linspace(1, 0, distill_steps + 1, device=device)[:, None, None]
        t_start_expanded = t_start[None, :, :]  # [1, batch_size, 1]
        t_end = torch.clamp(t_start_expanded - dt * distill_steps, min=eps)
        
        # Interpolate: [distill_steps+1, batch_size, 1]
        timesteps = t_start_expanded * space + (1 - space) * t_end
        
        # Initialize with corrupted input using t_start
        dalpha_t, alpha_t = self.noise(t_start)
        alpha_t = alpha_t.unsqueeze(-1)
        
        x = noisy_output['xt']  # Current state (continuous representation)
        clean_mask = noisy_output['reveal_mask'].bool()  # Which tokens are clean
        
        # Storage for teacher predictions
        teacher_predictions = torch.zeros(
            (batch_size, seq_len, self.backbone.dim), device=device
        )
        tokens_denoised = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
        embedding_cache = self.backbone.get_embedding(x)


        x = x.argmax(dim=-1)
        # Run teacher trajectory
        for i in range(distill_steps):
            t = timesteps[i].squeeze(-1)  # [batch_size]
            s = timesteps[i+1].squeeze(-1)  # [batch_size]
            
            # Get continuous noise for each batch element's timestep
            continuous_noise_t = self.get_continuous_from_discrete_noise(t)
            continuous_noise_s = self.get_continuous_from_discrete_noise(s)
            
            # Continuous step - get teacher's predictions
            embedding_cache, p_x0, output_embedding = self._continuous_step_teacher_embedding(x, 
                                                                 sigma_s = continuous_noise_t, 
                                                                 sigma_t=continuous_noise_s,
                                                                 time_s=s,
                                                                 time_t=t,
                                                                 embedding_cache=embedding_cache,
                                                                 clean_mask=clean_mask) 
            
            # sample x0 hat
            x0_hat = sample_categorical(p_x0)
            
            # Discrete step - sample updates  
            new_x, new_clean_mask = self._discrete_step_teacher(x0_hat, 
                                                                x, t, dt, 
                                                                prev_clean_mask=clean_mask)
            # Track which tokens got denoised this step
            newly_denoised = (~tokens_denoised) & (new_clean_mask.bool() & ~clean_mask.bool())


            
            # Store teacher predictions for newly denoised tokens
            if newly_denoised.any():
                teacher_predictions[newly_denoised] = output_embedding[newly_denoised].float()
            tokens_denoised = tokens_denoised | newly_denoised
            
            # Update state
            x = new_x
            clean_mask = new_clean_mask
        
        # For any remaining masked tokens, use final predictions
        still_masked = ~tokens_denoised
        if still_masked.any():
            teacher_predictions[still_masked] = output_embedding[still_masked].float()
        return teacher_predictions
               


    
    def nll(self, x0, output_tokens, current_accumulation_step=None, train_mode=False):
        del output_tokens
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        assert t.shape[0] == x0.shape[0]
        dalpha_t, alpha_t = self.noise(t)
        alpha_t = alpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2

        noisy_output = self.q_xt(x0, alpha_t)
        teacher_probs = self.sample_teacher_trajectories(x0, noisy_output, t.unsqueeze(-1), distill_steps=2)
        log_x_theta = self.forward(**noisy_output)
        kl_per_token = self._fwd_kl(log_x_theta, teacher_probs)
        utils.print_nans(log_x_theta, "model_output")

        return kl_per_token
    

    def _bwd_kl(self, preds, target):
        kl_per_coord = F.kl_div(preds, target, reduction="none", log_target=True)
        kl_per_position = kl_per_coord.sum(-1)
        return kl_per_position
    
    def _fwd_kl(self, preds, target):
        kl_per_coord = F.kl_div(target, preds, reduction="none", log_target=True)
        kl_per_position = kl_per_coord.sum(-1)
        return kl_per_position

    def sample_t_adaptive(self, batch_size, device):
        """
        Sample t ensuring we have enough runway for the teacher trajectory.
        Need: t_start >= dt * (distill_steps - 1) + eps
        """
        dt = self.dt 
        distill_steps = self.distill_steps
        min_t = dt * (distill_steps - 1) + self.sampling_eps
        
        if min_t >= 0.95:  # If we need almost the full range
            return torch.ones(batch_size, 1, device=device)
        else:
            return torch.rand(batch_size, 1, device=device) * (1.0 - min_t) + min_t
        
    def training_step(self, batch, batch_idx):
        step = self.global_step
        if step > 0 and step % self.adapt_every == 0 and step != self.last_update_step:
            self.update_round()
            self.update_teacher()
            self.last_update_step = step
            if self.number_steps <= self.max_distill_steps: 
                self.trainer.should_stop = True
        return super().training_step(batch, batch_idx)
    
def compute_growth_schedule(adapt_every, initial_steps, final_steps, total_training_steps):
    total_adaptations = total_training_steps // adapt_every
    total_compression = initial_steps / final_steps
    growth_factor = total_compression ** (1 / total_adaptations)
    return growth_factor, total_adaptations



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


def training_sigma(t: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
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


import torch
import numpy as np
from scipy.stats import norm

def compute_dsigma_dt(t, r_max, r_min):
    """
    Compute dσ/dt for the one-hot noise schedule.
    
    Given:
    - r(t) = (1 - t)(r_max - r_min) + r_min
    - σ(t) = 1 / (√2 · Φ⁻¹(r(t)))
    
    Returns dσ/dt where Φ⁻¹ is the inverse standard normal CDF.
    
    Args:
        t: Time vector, shape (n,) or scalar
        r_max: Maximum percentile probability (e.g., 0.99)
        r_min: Minimum percentile probability (e.g., 0.51)
    
    Returns:
        dsigma_dt: Derivative dσ/dt, same shape as t
    """
    # Convert to tensor if needed
    if isinstance(t, (int, float)):
        t = torch.tensor([t], dtype=torch.float32)
    elif not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32)
    
    # Compute r(t) = (1 - t)(r_max - r_min) + r_min
    r_t = (1 - t) * (r_max - r_min) + r_min
    
    # Compute dr/dt = -(r_max - r_min)
    dr_dt = -(r_max - r_min)
    
    # Compute Φ⁻¹(r(t)) - inverse standard normal CDF
    # Using scipy for numerical stability
    phi_inv_r = torch.tensor([norm.ppf(r_val.item()) for r_val in r_t], dtype=torch.float32)
    
    # Compute standard normal PDF at Φ⁻¹(r(t))
    # φ(x) = (1/√(2π)) * exp(-x²/2)
    phi_pdf = torch.exp(-0.5 * phi_inv_r**2) / torch.sqrt(torch.tensor(2 * np.pi))
    
    # Compute dσ/dr using chain rule
    # σ(r) = 1 / (√2 · Φ⁻¹(r))
    # dσ/dr = -1/(√2) · 1/(Φ⁻¹(r))² · d/dr[Φ⁻¹(r)]
    # d/dr[Φ⁻¹(r)] = 1/φ(Φ⁻¹(r))
    sqrt_2 = torch.sqrt(torch.tensor(2.0))
    dsigma_dr = -1 / (sqrt_2 * phi_inv_r**2 * phi_pdf)
    
    # Apply chain rule: dσ/dt = (dσ/dr) * (dr/dt)
    dsigma_dt = dsigma_dr * dr_dt
    
    return dsigma_dt

def compute_sigma(t, r_max, r_min):
    """
    Compute σ(t) for comparison.
    
    Args:
        t: Time vector, shape (n,) or scalar
        r_max: Maximum percentile probability
        r_min: Minimum percentile probability
    
    Returns:
        sigma: Noise level σ(t), same shape as t
    """
    if isinstance(t, (int, float)):
        t = torch.tensor([t], dtype=torch.float32)
    elif not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32)
    
    # Compute r(t)
    r_t = (1 - t) * (r_max - r_min) + r_min
    
    # Compute Φ⁻¹(r(t))
    phi_inv_r = torch.tensor([norm.ppf(r_val.item()) for r_val in r_t], dtype=torch.float32)
    
    # Compute σ(t) = 1 / (√2 · Φ⁻¹(r(t)))
    sqrt_2 = torch.sqrt(torch.tensor(2.0))
    sigma = 1 / (sqrt_2 * phi_inv_r)
    
    return sigma

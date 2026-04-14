from .uniform_state import UniformState
from .algo_utils import sample_categorical
from .registry import algo_registry

import fsspec
import pickle
import utils


import torch
import torch.nn.functional as F


import collections

@algo_registry.register("duo")
class DUO_BASE(UniformState):
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

        cur_logits = (self.forward(x, sigma_t)).double()
        xtheta_to_use = cur_logits.exp()



        q_xs = self._compute_posterior(
            x=xtheta_to_use, xt=x, alpha_s=alpha_s, alpha_t=alpha_t
        )
        if self.p_nucleus < 1:
            q_xs = utils.top_k_top_p_filtering(q_xs.log(), top_p=self.p_nucleus)

        xs = sample_categorical(q_xs)

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
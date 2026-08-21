"""RePlaid with masked-diffusion clean-token conditioning."""

import torch

from .registry import algo_registry
from .replaid import RePlaid


@algo_registry.register("replaid_keep")
class RePlaidKeep(RePlaid):
    """Use hard kept tokens in place of predicted self-conditioning."""

    default_sampler = "replaid_keep_ddpm"
    supported_samplers = frozenset(
        {"replaid_keep_ddpm", "replaid_keep_entropy_bounded"}
    )
    sampling_family = "replaid_keep"
    sampling_capabilities = frozenset(
        {
            "replaid_prior",
            "replaid_keep_denoise",
            "replaid_schedule",
            "embedding_projection",
        }
    )
    LOSS_VARIANTS = frozenset({"mse", "hybrid_elbo", "mse_reveal_ce"})

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.loss_variant = config.algo.loss_variant
        self.reveal_ce_weight = config.algo.reveal_ce_weight
        if self.loss_variant not in self.LOSS_VARIANTS:
            raise ValueError(
                f"loss_variant must be one of {sorted(self.LOSS_VARIANTS)}"
            )
        if config.algo.T <= 0:
            raise ValueError("RePlaidKeep requires a positive discrete T")
        if self.self_condition_rate != 0:
            raise ValueError(
                "RePlaidKeep replaces predicted self-conditioning; "
                "self_condition_rate must be zero"
            )
        if self.reveal_ce_weight < 0:
            raise ValueError("reveal_ce_weight must be nonnegative")

    @staticmethod
    def cadd_step_weights(gamma_t, gamma_s):
        """CADD equations 4.15–4.16 for a reverse step t -> s."""
        keep_t = torch.sigmoid(-gamma_t)
        keep_s = torch.sigmoid(-gamma_s)
        hidden_t = (1 - keep_t).clamp_min(1e-12)
        rho_flip = ((keep_s - keep_t) / hidden_t).clamp(0, 1)
        rho_keep = ((1 - keep_s) / hidden_t).clamp(0, 1)
        gamma_step = (keep_t / keep_s.clamp_min(1e-12)).clamp(0, 1)
        beta_tilde = (
            (1 - keep_s) * (1 - gamma_step) / hidden_t
        ).clamp_min(1e-12)
        coefficient = keep_s.sqrt() * (1 - gamma_step) / hidden_t
        mse_weight = coefficient.square() / (2 * beta_tilde)
        return rho_flip, rho_keep, mse_weight

    def _discrete_times(self, count, device):
        raw = self.sample_t(count, device)
        indices = (raw * self.config.algo.T).floor().long() + 1
        t = indices.to(torch.float64) / self.config.algo.T
        s = (indices - 1).to(torch.float64) / self.config.algo.T
        return t, s

    def nll(
        self,
        x0,
        output_tokens,
        current_accumulation_step=None,
        train_mode=False,
        valid_tokens=None,
    ):
        del output_tokens, current_accumulation_step
        batch_size, _ = x0.shape
        recon_size = self._split_size(batch_size)
        diffusion_size = batch_size - recon_size
        t = torch.zeros(batch_size, device=x0.device, dtype=torch.float64)
        s = torch.zeros_like(t)
        if diffusion_size:
            t_diff, s_diff = self._discrete_times(diffusion_size, x0.device)
            t[recon_size:], s[recon_size:] = t_diff, s_diff
        if train_mode:
            gamma, gamma_prime, variance_weight = self._training_gamma(
                t.float(), recon_size
            )
        else:
            gamma, gamma_prime = self._gamma(t.float())
            variance_weight = None
        gamma_s, _ = self._gamma(s.float())

        clean = self.backbone.embed_tokens(x0)
        noisy = self.q_sample(clean, gamma)
        keep_probability = torch.sigmoid(-gamma).to(clean)[:, None]
        keep = torch.rand_like(x0.float()) < keep_probability
        keep[:recon_size] = False
        if valid_tokens is not None:
            keep |= ~valid_tokens.bool()
        context = torch.where(keep.unsqueeze(-1), clean, torch.zeros_like(clean))
        model_input = torch.where(keep.unsqueeze(-1), clean, noisy)
        log_probs = self.denoise(model_input, gamma.float(), context)
        predicted = self.backbone.embed_tokens(log_probs.exp())
        ce = -log_probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        squared_error = (predicted - clean).square().sum(-1)
        hidden = ~keep

        snr_prime = -gamma_prime * torch.exp(-gamma)
        plaid_mse = -0.5 * snr_prime.to(clean)[:, None] * squared_error
        rho_flip, rho_keep, cadd_mse_weight = self.cadd_step_weights(
            gamma, gamma_s
        )
        if self.loss_variant == "hybrid_elbo":
            # Uniformly sampling one of T discrete reverse transitions requires
            # multiplying its KL by T to estimate their ELBO sum.
            diffusion = self.config.algo.T * (
                rho_flip.to(clean)[:, None] * ce
                + rho_keep.to(clean)[:, None]
                * cadd_mse_weight.to(clean)[:, None]
                * squared_error
            )
        elif self.loss_variant == "mse_reveal_ce":
            diffusion = plaid_mse + (
                self.reveal_ce_weight
                * self.config.algo.T
                * rho_flip.to(clean)[:, None]
                * ce
            )
        else:
            diffusion = plaid_mse
        diffusion = diffusion * hidden.to(diffusion)

        if variance_weight is not None:
            variance_weight[0] = torch.cat(
                (
                    torch.zeros(recon_size, device=x0.device, dtype=torch.float64),
                    2 * diffusion[recon_size:].mean(-1).detach().double(),
                )
            )
        gamma1, _ = self._gamma(torch.ones(1, device=x0.device))
        alpha1_sq = torch.sigmoid(-gamma1[0])
        sigma1_sq = 1 - alpha1_sq
        prior = 0.5 * (
            alpha1_sq.to(clean) * clean.square().sum(-1)
            + self.config.algo.embedding_dim
            * (sigma1_sq - 1 - torch.log(sigma1_sq)).to(clean)
        )
        losses = prior
        if recon_size:
            losses[:recon_size] += ce[:recon_size] * batch_size / recon_size
        if diffusion_size:
            losses[recon_size:] += (
                diffusion[recon_size:] * batch_size / diffusion_size
            )
        if train_mode and recon_size and diffusion_size:
            momentum = self.split_ema
            recon_std = ce[:recon_size].mean(-1).std(unbiased=False)
            diff_std = diffusion[recon_size:].mean(-1).std(unbiased=False)
            self.recon_std_ema.lerp_(
                recon_std.detach().to(self.recon_std_ema), 1 - momentum
            )
            self.diff_std_ema.lerp_(
                diff_std.detach().to(self.diff_std_ema), 1 - momentum
            )
        self._record_time_loss(t.float(), losses, valid_tokens, train_mode)
        return losses

    @torch.no_grad()
    def sample_replaid_keep_denoise(self, z_t, t, kept_tokens, kept_mask):
        gamma = self.sampling_gamma(t)
        clean_context = self.sampling_embeddings(kept_tokens)
        clean_context = torch.where(
            kept_mask.unsqueeze(-1), clean_context, torch.zeros_like(clean_context)
        )
        log_probs = self.denoise(z_t, gamma.float(), clean_context)
        predicted = self.sampling_embeddings(log_probs.exp())
        return log_probs, predicted, gamma

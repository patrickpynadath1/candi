"""CANDI with the SJD paper's cross-position Gaussian blending kernel."""

import torch
import torch.nn.functional as F

from .candi import CANDI
from .neighbor_blurring import blend_positions, gaussian_position_kernel
from .registry import algo_registry


@algo_registry.register("candi_ra_change")
class CANDIRAChange(CANDI):
    """Spatially smoothed CANDI (the identity limit is ordinary CANDI)."""

    default_sampler = "candi"
    supported_samplers = frozenset({"candi"})

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        spatial = config.algo.get("spatial_blurring", {})
        self.spatial_blurring_bandwidth = spatial.get(
            "bandwidth", spatial.get("sigma", 0.0)
        )
        self.spatial_blurring_include_self = spatial.get("include_self", True)
        kernel = gaussian_position_kernel(
            self.num_tokens,
            self.spatial_blurring_bandwidth,
            include_self=self.spatial_blurring_include_self,
        )
        self.register_buffer("spatial_kernel", kernel, persistent=True)

    def blend_positions(self, values):
        return blend_positions(values, self.spatial_kernel)

    def q_xt(
        self, x, alpha_t, valid_tokens=None, continuous_discrete_noise=None
    ):
        disc_xt = self.discrete_noising(x, alpha_t)
        if valid_tokens is not None:
            disc_xt = torch.where(valid_tokens.bool(), disc_xt, x)
        reveal_mask = (disc_xt == x).float()

        discrete_noise = (1 - alpha_t).reshape(-1)
        rank_noise = (
            discrete_noise
            if continuous_discrete_noise is None
            else continuous_discrete_noise.reshape(-1)
        )
        continuous_noise = self.get_continuous_from_discrete_noise(rank_noise)

        onehot = F.one_hot(x, num_classes=self.vocab_size - 1).float()
        blended_mean = self.blend_positions(onehot)
        noisy = blended_mean + continuous_noise[:, None, None].to(x.device) * (
            torch.randn_like(onehot)
        )
        xt = (
            onehot * reveal_mask.unsqueeze(-1)
            + noisy * (1 - reveal_mask).unsqueeze(-1)
        )
        return {
            "xt": xt,
            "reveal_mask": reveal_mask,
            "discrete_noise": discrete_noise,
            "rank_noise": rank_noise,
            "continuous_noise": continuous_noise,
        }

    @torch.no_grad()
    def _continuous_step(
        self,
        x,
        time_t,
        time_s,
        sigma_s,
        sigma_t,
        clean_mask=None,
        is_embed=False,
    ):
        del time_s, is_embed
        dt = torch.ones(x.shape[0], device=x.device) * (sigma_s - sigma_t).item()
        time = torch.ones(x.shape[0], device=x.device) * time_t.item()
        sigma = torch.ones(x.shape[0], device=x.device) * sigma_t.item()
        if clean_mask is None:
            clean_mask = torch.zeros(x.shape[:-1], device=x.device)
        log_probs = self.forward(
            xt=x,
            discrete_noise=time,
            reveal_mask=clean_mask,
            continuous_noise=sigma,
        ).double()
        denoised = log_probs.exp()
        blended_posterior_mean = self.blend_positions(denoised)
        score_residual = (x - blended_posterior_mean) / (
            sigma[:, None, None] ** 2
        )
        return x - dt[:, None, None] * score_residual, denoised

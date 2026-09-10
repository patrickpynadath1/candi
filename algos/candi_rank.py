"""Paper-normalized CANDI rank coupling; legacy CANDI stays unchanged."""

import torch

from .candi import CANDI
from .rank_schedules import geometric_rank, sigma_from_rank, onehot_rank
from .registry import algo_registry


@algo_registry.register("candi_rank")
class CANDIRank(CANDI):
    default_sampler = "candi"
    supported_samplers = frozenset({"candi"})
    exact_time_endpoints = True
    conventional_sigma_order = True

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        if self.factorized_adaptive_schedule or config.algo.curve_type != "linear":
            raise ValueError("candi_rank currently requires a single linear corruption path")
        if self.cont_schedule_target != "r":
            raise ValueError("candi_rank requires cont_schedule_target=r")
        if self.is_embed:
            raise ValueError("use candi_embedding for embedding-space corruption")
        if not self.use_percentile_scheduling:
            raise ValueError("candi_rank requires the same rank schedule at inference")
        if (config.noise.eps != 0 or config.noise.get("alpha_min") is not None
                or config.noise.get("alpha_max") is not None):
            raise ValueError("candi_rank requires untruncated noise with eps=0")
        geometric_rank(torch.tensor(0.), self.min_percentile, self.max_percentile)
        self.get_continuous_from_discrete_noise(torch.tensor([0., 1.]))

    def get_continuous_from_discrete_noise(self, discrete_noise):
        if self.cont_schedule_type == "linear":
            rank = self.min_percentile + discrete_noise * (
                self.max_percentile - self.min_percentile
            )
        elif self.cont_schedule_type == "geometric":
            rank = geometric_rank(discrete_noise, self.min_percentile, self.max_percentile)
        else:
            raise ValueError("candi_rank supports linear and geometric rank coupling")
        return sigma_from_rank(rank)

    def discrete_noising(self, x, alpha_t):
        # A sentinel represents corruption; no accidental identity retention.
        keep = torch.zeros_like(x, dtype=torch.bool) if self.pure_continuous else (
            torch.rand_like(x, dtype=torch.float32) < alpha_t
        )
        if self.ignore_bos:
            keep[:, 0] = True
        return torch.where(keep, x, self.mask_index)

    def prior_sample(self, *batch_dims):
        sigma = self.get_continuous_from_discrete_noise(torch.ones((), device=self.device))
        # Finite-sigma Gaussian prior approximation, as in CANDI.
        return torch.randn(*batch_dims, self.vocab_size-1, device=self.device) * sigma

    @torch.no_grad()
    def rank_diagnostics(self):
        t = torch.linspace(0, 1, 101, device=self.device)
        rho = 1-self.noise.alpha_t(t)
        sigma = self.get_continuous_from_discrete_noise(rho)
        return dict(clock=t.tolist(), progress=rho.tolist(), sigma=sigma.tolist(),
                    rank=onehot_rank(sigma).tolist(), space="onehot",
                    rank_condition="continuously corrupted positions",
                    prior="finite-sigma Gaussian approximation")

    @torch.no_grad()
    def _continuous_step(self, x, time_t, time_s, sigma_s, sigma_t,
                         clean_mask=None, is_embed=False, **kwargs):
        del time_s, is_embed, kwargs
        if clean_mask is None:
            clean_mask = torch.zeros(x.shape[:2], device=x.device, dtype=torch.bool)
        probs = self.forward(
            xt=x, discrete_noise=time_t.expand(x.shape[0]),
            continuous_noise=sigma_t.expand(x.shape[0]), reveal_mask=clean_mask,
        ).exp().to(x.dtype)
        # Deterministic Gaussian denoised-estimate transition. Exact for a
        # constant clean prediction and first-order consistent with VE PF-ODE.
        result = probs + (sigma_s / sigma_t) * (x - probs)
        return torch.where(clean_mask.bool().unsqueeze(-1), x, result), probs

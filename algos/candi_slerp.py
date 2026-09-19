"""CANDI masking coupled to dimension-dependent spherical rank degradation."""

import torch

from .candi_embedding import CANDIEmbedding
from .rank_schedules import SLERPRankSchedule
from .registry import algo_registry
from .spherical_utils import (
    exp_map,
    slerp,
    sphere_normalize,
    spherical_truncation_alpha_max,
)
from samplers.sfm import sfm_compute_velocity


@algo_registry.register("candi_slerp")
class CANDISLERP(CANDIEmbedding):
    # Reuse the hybrid reveal process; the transition itself is spherical.
    make_rank_schedule = staticmethod(SLERPRankSchedule)
    supported_interpolations = {"geometric_rank", "truncated_linear"}

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        if self.interpolation_schedule == "truncated_linear":
            delta = config.algo.truncation_delta
            self.clean_alpha_max = spherical_truncation_alpha_max(
                self.vocab_size,
                self.backbone.input_dim,
                delta,
            )
            self.clean_alpha_min = float(
                config.algo.get("truncation_alpha_min", 0.0)
            )
            if not 0 <= self.clean_alpha_min < self.clean_alpha_max:
                raise ValueError(
                    "truncation_alpha_min must be nonnegative and below "
                    "the spherical truncation alpha_max"
                )
            self.rank_schedule = self.make_rank_schedule(self.backbone.input_dim)
        self.slerp_precision = config.algo.slerp_precision
        if self.slerp_precision not in {"float32", "float64"}:
            raise ValueError("slerp_precision must be float32 or float64")

    def interpolation(self, progress):
        if self.interpolation_schedule == "truncated_linear":
            return 1 - (
                self.clean_alpha_min
                + (self.clean_alpha_max - self.clean_alpha_min) * (1 - progress)
            )
        return super().interpolation(progress)

    def prior_sample(self, *batch_dims):
        return sphere_normalize(super().prior_sample(*batch_dims))

    def q_xt(self, x, alpha_t, valid_tokens=None):
        rho = (1 - alpha_t).reshape(-1)
        u = self.interpolation(rho)
        clean = self.backbone.get_embedding(x)
        noise = sphere_normalize(torch.randn_like(clean))
        if self.slerp_precision == "float64":
            xt = slerp(clean.double(), noise.double(), u.double()).to(clean.dtype)
        else:
            xt = slerp(clean, noise, u)
        keep = torch.zeros_like(x, dtype=torch.bool)
        if self.hybrid_mask:
            keep = torch.rand_like(x, dtype=torch.float32) < alpha_t
        if valid_tokens is not None:
            keep |= ~valid_tokens.bool()
        xt = torch.where(keep[..., None], clean, xt)
        return dict(
            xt=xt,
            discrete_noise=rho,
            continuous_noise=u,
            reveal_mask=keep,
            clean_tokens=x,
        )

    @torch.no_grad()
    def sample_continuous_transition(
        self, x, time_t, time_s, sigma_t, sigma_s, clean_mask=None, clean_tokens=None
    ):
        del time_s
        probs = self.forward(
            xt=x,
            discrete_noise=time_t.expand(len(x)),
            continuous_noise=sigma_t.expand(len(x)),
            reveal_mask=clean_mask,
            clean_tokens=clean_tokens,
        ).exp()
        table = self.backbone.get_embedding(
            torch.arange(self.vocab_size, device=x.device)[None]
        )[0]
        geometry = x.double() if self.slerp_precision == "float64" else x.float()
        velocity = sfm_compute_velocity(
            geometry, table.to(geometry), probs.to(geometry).log()
        )
        # u measures remaining angular distance to a fixed clean endpoint.
        step = (sigma_t - sigma_s) / sigma_t.clamp_min(1e-12)
        state = exp_map(geometry, step * velocity).to(x.dtype)
        state = self.sampling_prompt_state(state, clean_tokens, clean_mask)
        return state, probs

    @torch.no_grad()
    def rank_diagnostics(self):
        t = torch.linspace(0, 1, 101, device=self.device)
        rho = 1 - self.noise.alpha_t(t)
        u = self.interpolation(rho)
        return dict(
            clock=t.tolist(),
            progress=rho.tolist(),
            interpolation=u.tolist(),
            rank=self.rank_schedule.expected_rank(u).tolist(),
            embedding_dimension=int(self.rank_schedule.dimension),
            space="spherical_embeddings",
            calibration="iid uniform sphere",
            rank_condition="continuously corrupted positions",
            prior="uniform sphere",
            prior_matches_endpoint=(self.interpolation_schedule == "truncated_linear" or self.rank_schedule.endpoint_width > 0),
            interpolation_schedule=self.interpolation_schedule,
            endpoint_width=self.rank_schedule.endpoint_width,
            truncation_alpha_min=getattr(self, "clean_alpha_min", None),
            truncation_alpha_max=getattr(self, "clean_alpha_max", None),
        )

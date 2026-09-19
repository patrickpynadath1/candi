"""Hybrid discrete masking with Gaussian interpolation in unit embeddings."""

import torch

from .candi import (
    HYBRID_LOSS_WEIGHTINGS,
    adaptive_observation_mask,
    hybrid_token_loss,
)
from .cont_diff_embed import ContDiffEmbed
from .rank_schedules import SphereRankSchedule, embedding_table_rank
from .registry import algo_registry


@algo_registry.register("candi_embedding")
class CANDIEmbedding(ContDiffEmbed):
    sampling_family = "candi_embedding"
    default_sampler = "candi_embedding"
    supported_samplers = frozenset({
        "candi_embedding", "candi_embedding_entropy_bounded"
    })
    exact_time_endpoints = True

    supported_interpolations = {"linear", "linear_rank", "geometric_rank"}

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.hybrid_mask = not config.algo.pure_continuous
        self.hybrid_loss_weighting = config.algo.get(
            "hybrid_loss_weighting", "none"
        )
        if self.hybrid_loss_weighting not in HYBRID_LOSS_WEIGHTINGS:
            raise ValueError(
                "algo.hybrid_loss_weighting must be 'none' or 'likelihood'"
            )
        self.interpolation_schedule = config.algo.interpolation_schedule
        if config.algo.get("factorized_adaptive_schedule", False):
            raise ValueError("candi_embedding uses one coupled time axis, not factorized times")
        if self.interpolation_schedule not in self.supported_interpolations:
            raise ValueError(f"unsupported interpolation_schedule: {self.interpolation_schedule}")
        if config.algo.continuous_path != "flow_matching":
            raise ValueError("candi_embedding requires flow_matching interpolation")
        if (config.noise.eps != 0 or config.noise.get("alpha_min") is not None
                or config.noise.get("alpha_max") is not None):
            raise ValueError("candi_embedding requires untruncated noise with eps=0")
        if self.interpolation_schedule in {"linear_rank", "geometric_rank"}:
            self.rank_schedule = self.make_rank_schedule(
                self.backbone.input_dim, config.algo.min_percentile,
                config.algo.max_percentile, config.algo.rank_endpoint_width,
            )

    make_rank_schedule = staticmethod(SphereRankSchedule)

    def interpolation(self, progress):
        if self.interpolation_schedule in {"linear_rank", "geometric_rank"}:
            return self.rank_schedule(progress, coupling="linear" if self.interpolation_schedule == "linear_rank" else "geometric")
        return progress

    def clean_state(self, x):
        return self.backbone.get_embedding(x)

    def q_xt(self, x, alpha_t, valid_tokens=None):
        progress = (1-alpha_t).reshape(-1)
        u = self.interpolation(progress)
        embeddings = self.clean_state(x)
        xt = (1-u[:, None, None])*embeddings + u[:, None, None]*torch.randn_like(embeddings)
        keep = torch.zeros_like(x, dtype=torch.bool)
        if self.hybrid_mask:
            keep = torch.rand_like(x, dtype=torch.float32) < alpha_t
        if valid_tokens is not None:
            keep |= ~valid_tokens.bool()
        xt = torch.where(keep.unsqueeze(-1), embeddings, xt)
        return dict(xt=xt, discrete_noise=progress, continuous_noise=u,
                    reveal_mask=keep, clean_tokens=x)

    def forward(self, xt, discrete_noise, continuous_noise=None,
                reveal_mask=None, clean_tokens=None, **kwargs):
        del kwargs
        logits = self.backbone(
            xt=xt, discrete_noise=discrete_noise,
            continuous_noise=continuous_noise, reveal_mask=None,
            is_embed=self.config.algo.is_embed, precond=False,
        )
        log_probs = self._process_model_output(logits)
        if reveal_mask is not None:
            fixed = torch.full_like(log_probs, self.neg_infinity)
            fixed.scatter_(-1, clean_tokens.unsqueeze(-1), 0)
            log_probs = torch.where(reveal_mask.unsqueeze(-1), fixed, log_probs)
        return log_probs

    def nll(self, x0, output_tokens, current_accumulation_step=None,
            train_mode=False, valid_tokens=None):
        del output_tokens
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        derivative, alpha = self.noise(t)
        noisy = self.q_xt(x0, alpha[:, None], valid_tokens)
        log_probs = self.forward(**noisy)
        raw_ce = -log_probs.gather(-1, x0[..., None]).squeeze(-1)
        loss = raw_ce
        if self.hybrid_mask:
            loss = hybrid_token_loss(
                raw_ce, derivative, alpha, self.hybrid_loss_weighting
            )
        adaptive_valid_tokens = adaptive_observation_mask(
            valid_tokens, noisy["reveal_mask"], not self.hybrid_mask
        )
        self._record_time_loss(
            t, raw_ce, adaptive_valid_tokens, train_mode
        )
        return loss

    def sampling_continuous_schedule(self, timesteps):
        return self.interpolation(1-self.noise.alpha_t(timesteps))

    @torch.no_grad()
    def rank_diagnostics(self):
        t = torch.linspace(0, 1, 101, device=self.device)
        rho = 1-self.noise.alpha_t(t)
        u = self.interpolation(rho)
        table = self.backbone.get_embedding(torch.arange(self.vocab_size, device=self.device)[None])[0]
        return dict(clock=t.tolist(), progress=rho.tolist(), interpolation=u.tolist(),
                    actual_table_rank=embedding_table_rank(table, u).tolist(),
                    space="unit_embeddings", rank_condition="continuously corrupted positions",
                    calibration="iid uniform sphere" if self.interpolation_schedule in {"linear_rank", "geometric_rank"} else "none",
                    prior="standard Gaussian")

    def sampling_prompt_state(self, state, tokens, mask):
        if tokens is None:
            return state
        return torch.where(mask.unsqueeze(-1), self.clean_state(tokens), state)

    @torch.no_grad()
    def sample_continuous_transition(self, x, time_t, time_s, sigma_t,
                                     sigma_s, clean_mask=None, clean_tokens=None):
        del time_s
        probs = self.forward(
            xt=x, discrete_noise=time_t.expand(len(x)),
            continuous_noise=sigma_t.expand(len(x)), reveal_mask=clean_mask,
            clean_tokens=clean_tokens,
        ).exp()
        clean_estimate = self.clean_state(probs).to(x.dtype)
        # sigma here is interpolation u, not Gaussian standard deviation.
        # Integrating with delta-u incorporates the clock warp exactly for a
        # frozen clean prediction; no second warp derivative belongs here.
        next_x = clean_estimate + sigma_s/sigma_t.clamp_min(1e-12)*(x-clean_estimate)
        return next_x, probs

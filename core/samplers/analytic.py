"""Analytic absorbing reverse steps for SEDD scores."""

import torch
from .discrete import DiscreteSampler
from core.parameterizations.score import AbsorbingScoreParameterization
from core.processes.absorbing import AbsorbingProcess
from .utils import initialize_state


class AbsorbingAnalyticUpdate:
    prediction_feedback = False

    def __init__(
        self, steps=1024, epsilon=1e-3, final_denoise="ancestral", use_float64=True
    ):
        if not isinstance(steps, int) or steps <= 0 or not 0 < epsilon < 1:
            raise ValueError("positive steps and epsilon in (0,1) required")
        if final_denoise not in {"ancestral", "none"}:
            raise ValueError(
                "score analytic sampling supports ancestral or no final denoising"
            )
        self.steps, self.epsilon, self.final_denoise = steps, epsilon, final_denoise
        self.use_float64 = use_float64

    def _scores(self, log_scores):
        # Legacy exponentiates first, then promotes transition calculations.
        scores = log_scores.exp()
        return scores.double() if self.use_float64 else scores

    def initialize(self, algo, prompt_tokens, prompt_mask):
        if not isinstance(
            algo.predictor, AbsorbingScoreParameterization
        ) or not isinstance(algo.process, AbsorbingProcess):
            raise TypeError("analytic sampling requires absorbing scores and process")
        return initialize_state(
            algo,
            prompt_tokens,
            prompt_mask,
            predictor_type=AbsorbingScoreParameterization,
        )

    def update(self, algo, state, time, next_time, prompt_tokens, prompt_mask):
        keep, _ = algo.schedule(time.expand(len(state)))
        next_keep, _ = algo.schedule(next_time.expand(len(state)))
        scores = self._scores(algo.sampling_prediction(state, keep))
        state = algo.process.analytic_reverse(
            state, scores, next_keep.log() - keep.log()
        )
        return torch.where(prompt_mask.bool(), prompt_tokens, state), 1

    def finalize(self, algo, state, time, prompt_tokens, prompt_mask):
        if self.final_denoise == "none":
            return state, 0
        keep, _ = algo.schedule(time.expand(len(state)))
        scores = self._scores(algo.sampling_prediction(state, keep))
        state = algo.process.analytic_reverse(state, scores, -keep.log(), denoise=True)
        return torch.where(prompt_mask.bool(), prompt_tokens, state), 1


class AnalyticSampler(DiscreteSampler):
    def __init__(
        self,
        steps=1024,
        epsilon=1e-3,
        final_denoise="ancestral",
        use_float64=True,
        *,
        update_rule=None,
        time_grid=None,
    ):
        rule = update_rule or AbsorbingAnalyticUpdate(
            steps, epsilon, final_denoise, use_float64
        )
        super().__init__(rule, steps, epsilon, time_grid)
        self.final_denoise, self.use_float64 = final_denoise, use_float64

    def _scores(self, log_scores):
        return self.update_rule._scores(log_scores)

    @torch.no_grad()
    def sample(self, algo, prompt_tokens, prompt_mask):
        return super().sample(algo, prompt_tokens, prompt_mask)

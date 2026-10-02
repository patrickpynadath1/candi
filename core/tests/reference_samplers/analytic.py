"""Analytic absorbing reverse steps for SEDD scores."""

import torch
from core.parameterizations.score import AbsorbingScoreParameterization
from core.processes.absorbing import AbsorbingProcess
from core.samplers.utils import initialize_state


class AnalyticSampler:
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

    @torch.no_grad()
    def sample(self, algo, prompt_tokens, prompt_mask):
        if not isinstance(
            algo.predictor, AbsorbingScoreParameterization
        ) or not isinstance(algo.process, AbsorbingProcess):
            raise TypeError("analytic sampling requires absorbing scores and process")
        state = initialize_state(
            algo,
            prompt_tokens,
            prompt_mask,
            predictor_type=AbsorbingScoreParameterization,
        )
        times = torch.linspace(1, self.epsilon, self.steps + 1, device=state.device)
        for time, next_time in zip(times[:-1], times[1:]):
            keep, _ = algo.schedule(time.expand(len(state)))
            next_keep, _ = algo.schedule(next_time.expand(len(state)))
            scores = self._scores(algo.sampling_prediction(state, keep))
            state = algo.process.analytic_reverse(
                state, scores, next_keep.log() - keep.log()
            )
            state = torch.where(prompt_mask.bool(), prompt_tokens, state)
        nfe = self.steps
        if self.final_denoise == "ancestral":
            keep, _ = algo.schedule(times[-1].expand(len(state)))
            scores = self._scores(algo.sampling_prediction(state, keep))
            state = algo.process.analytic_reverse(
                state, scores, -keep.log(), denoise=True
            )
            state = torch.where(prompt_mask.bool(), prompt_tokens, state)
            nfe += 1
        return state, {"nfe": nfe, "nfe_per_sample": [nfe] * len(state)}

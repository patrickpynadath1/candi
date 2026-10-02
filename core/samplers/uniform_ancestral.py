"""Ancestral uniform sampling with matched plug-in or posterior transitions."""

import torch
from core.parameterizations.plug_in import PlugInParameterization
from core.processes.uniform import UniformProcess
from core.processes.uniform_posterior import PosteriorUniformProcess
from core.parameterizations.posterior import PosteriorParameterization
from core.parameterizations.constraints import UnconstrainedPosterior
from core.modifiers.self_conditioning.session import prediction_session


def initialize_uniform(algo, prompt_tokens, prompt_mask):
    posterior_process = isinstance(algo.process, PosteriorUniformProcess)
    valid_predictor = (
        (
            isinstance(algo.predictor, PosteriorParameterization)
            and isinstance(algo.predictor.constraints, UnconstrainedPosterior)
            and algo.predictor.mask_index is None
        )
        if posterior_process
        else isinstance(algo.predictor, PlugInParameterization)
    )
    if not isinstance(algo.process, UniformProcess) or not valid_predictor:
        raise TypeError(
            "uniform samplers require matching posterior/process or plug-in/process semantics"
        )
    if (
        prompt_tokens.ndim != 2
        or prompt_mask.shape != prompt_tokens.shape
        or not len(prompt_tokens)
    ):
        raise ValueError("expected matching nonempty [batch, length] prompts and masks")
    state = algo.process.prior(prompt_tokens.shape, prompt_tokens.device)
    return torch.where(prompt_mask.bool(), prompt_tokens, state)


class UniformAncestralSampler:
    def __init__(
        self, steps=1024, epsilon=1e-3, final_denoise="ancestral", use_float64=True
    ):
        if not isinstance(steps, int) or steps <= 0 or not 0 < epsilon < 1:
            raise ValueError("positive steps and epsilon in (0,1) required")
        if final_denoise not in {"ancestral", "greedy", "none"}:
            raise ValueError("invalid final_denoise")
        self.steps, self.epsilon, self.final_denoise = steps, epsilon, final_denoise
        self.use_float64 = use_float64

    def _probabilities(self, prediction):
        return (prediction.double() if self.use_float64 else prediction.float()).exp()

    @torch.no_grad()
    def sample(self, algo, prompt_tokens, prompt_mask):
        algo = prediction_session(algo)
        state = initialize_uniform(algo, prompt_tokens, prompt_mask)
        times = torch.linspace(1, self.epsilon, self.steps + 1, device=state.device)
        for time, next_time in zip(times[:-1], times[1:]):
            keep, _ = algo.schedule(time.expand(len(state)))
            next_keep, _ = algo.schedule(next_time.expand(len(state)))
            prediction = algo.sampling_prediction(state, keep)
            state = algo.process.reverse(
                state, self._probabilities(prediction), keep, next_keep
            )
            state = torch.where(prompt_mask.bool(), prompt_tokens, state)
        nfe = self.steps
        if self.final_denoise != "none":
            keep, _ = algo.schedule(times[-1].expand(len(state)))
            if self.final_denoise == "greedy":
                state = algo.greedy_prediction(state, keep).argmax(-1)
            else:
                prediction = algo.sampling_prediction(state, keep)
                state = algo.process.reverse(
                    state, self._probabilities(prediction), keep, torch.ones_like(keep)
                )
            state = torch.where(prompt_mask.bool(), prompt_tokens, state)
            nfe += 1
        return state, {"nfe": nfe, "nfe_per_sample": [nfe] * len(state)}

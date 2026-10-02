"""Stochastic reverse Gaussian bridge sampler for blurring diffusion."""

import torch
from .loop import FixedGridLoop
from core.sequences import resolve_sequence_layout
from core.modifiers.self_conditioning.session import (
    PredictionSession,
    prediction_session,
)
from core.parameterizations.posterior import PosteriorParameterization


class BlurAncestralSampler:
    def __init__(
        self, steps=64, greedy_clean_estimate=False, self_conditioning_mode="previous"
    ):
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if self_conditioning_mode not in {"previous", "bootstrap"}:
            raise ValueError("self_conditioning_mode must be previous or bootstrap")
        self.steps = steps
        self.greedy_clean_estimate = bool(greedy_clean_estimate)
        self.self_conditioning_mode = self_conditioning_mode

    @torch.no_grad()
    def sample(self, algo, prompt_tokens, prompt_mask):
        if not isinstance(algo.predictor, PosteriorParameterization):
            raise TypeError("blur ancestral sampling requires a clean-token posterior")
        algo = prediction_session(algo)
        clean_prompt = algo.representation.encode(prompt_tokens)
        active = ~prompt_mask.bool()
        layout = resolve_sequence_layout(active)
        state = algo.process.prior(clean_prompt)
        state = torch.where(prompt_mask.bool()[..., None], clean_prompt, state)
        times = torch.linspace(
            1, 0, self.steps + 1, device=state.device, dtype=state.dtype
        )
        nfe = 0

        def predict(value, time):
            nonlocal nfe
            if (
                isinstance(algo, PredictionSession)
                and self.self_conditioning_mode == "bootstrap"
            ):
                algo.state.conditioning = None
                algo.sampling_prediction(value, time)
                nfe += 1
            result = algo.sampling_prediction(value, time)
            nfe += 1
            return result

        def update(state, current, target, index):
            time = current.expand(len(state))
            next_time = target.expand(len(state))
            probabilities = predict(state, time).float().exp()
            estimate = (
                algo.representation.encode(probabilities.argmax(-1))
                if self.greedy_clean_estimate
                else algo.representation.posterior_mean(probabilities)
            ).to(state)
            current_path = algo.schedule.for_layout(
                time, active, state.shape[-1], layout
            )
            target_path = algo.schedule.for_layout(
                next_time, active, state.shape[-1], layout
            )
            state = algo.process.reverse(state, estimate, current_path, target_path)
            state = torch.where(prompt_mask.bool()[..., None], clean_prompt, state)
            return state

        state = FixedGridLoop()(state, times, update)
        final_time = torch.zeros(len(state), device=state.device, dtype=state.dtype)
        final = predict(state, final_time)
        tokens = torch.where(prompt_mask.bool(), prompt_tokens, final.argmax(-1))
        return tokens, {"nfe": nfe, "nfe_per_sample": [nfe] * len(tokens)}

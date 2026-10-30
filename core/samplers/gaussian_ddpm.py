"""Gaussian reverse-posterior sampling with local prediction feedback."""

import math
import torch
from .loop import FixedGridLoop

from core.parameterizations.posterior import PosteriorParameterization
from core.modifiers.self_conditioning.session import (
    PredictionSession,
    prediction_session,
)


class GaussianDDPMSampler:
    def __init__(
        self,
        steps=180,
        epsilon=1e-5,
        prior_mode="standard_normal",
        self_conditioning_mode="previous",
        prompt_mode="noisy",
        final_transition_noise=False,
    ):
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if not math.isfinite(epsilon) or not 0 <= epsilon < 1:
            raise ValueError("epsilon must lie in [0, 1)")
        if prior_mode not in {"standard_normal", "schedule"}:
            raise ValueError("prior_mode must be standard_normal or schedule")
        if self_conditioning_mode not in {"previous", "bootstrap"}:
            raise ValueError("self_conditioning_mode must be previous or bootstrap")
        if prompt_mode not in {"clean", "noisy"}:
            raise ValueError("prompt_mode must be clean or noisy")
        self.self_conditioning_mode = self_conditioning_mode
        self.prompt_mode = prompt_mode
        self.final_transition_noise = final_transition_noise
        self.prior_mode = prior_mode
        self.steps, self.epsilon = steps, epsilon

    def steps_for_nfe(self, nfe):
        """Reserve terminal prediction and account for per-step SC bootstraps."""
        if isinstance(nfe, bool) or int(nfe) != nfe:
            raise ValueError("NFE budget must be an integer")
        calls = 2 if self.self_conditioning_mode == "bootstrap" else 1
        steps = int(nfe) // calls - 1
        if steps < 1:
            raise ValueError("NFE budget cannot fit a transition and final prediction")
        return steps

    @torch.no_grad()
    def sample(self, algo, prompt_tokens, prompt_mask):
        if not isinstance(algo.predictor, PosteriorParameterization):
            raise TypeError("Gaussian DDPM requires a clean-token posterior")
        algo = prediction_session(algo)
        clean = algo.representation.encode(prompt_tokens)
        initial_time = torch.ones(len(clean), device=clean.device)
        prior_sigma = (
            torch.ones_like(initial_time)
            if self.prior_mode == "standard_normal"
            else algo.schedule(initial_time).sigma
        )
        state = algo.process.prior(clean, prior_sigma)
        prompt_noise = torch.randn_like(clean) if prompt_mask.any() else None
        fixed = prompt_mask.bool()[..., None]
        times = torch.linspace(1, self.epsilon, self.steps + 1, device=state.device)

        def clamp_prompt(value, path):
            if prompt_noise is None:
                return value
            if self.prompt_mode == "clean":
                return torch.where(fixed, clean, value)
            prompt = (
                path.alpha.to(clean)[:, None, None] * clean
                + path.sigma.to(clean)[:, None, None] * prompt_noise
            )
            return torch.where(fixed, prompt, value)

        nfe = 0

        def preserve_feedback():
            if (
                isinstance(algo, PredictionSession)
                and algo.prediction.source == "posterior_mean"
            ):
                algo.state.conditioning = torch.where(
                    fixed, clean, algo.state.conditioning
                )

        def predict(value, time, path):
            nonlocal nfe
            if (
                isinstance(algo, PredictionSession)
                and self.self_conditioning_mode == "bootstrap"
            ):
                algo.state.conditioning = None
                algo.sampling_prediction(value, time, path)
                nfe += 1
                preserve_feedback()
            result = algo.sampling_prediction(value, time, path)
            nfe += 1
            preserve_feedback()
            return result

        path = algo.schedule(times[0].expand(len(state)))
        state = clamp_prompt(state, path)

        def update(state, time, next_time, index):
            time = time.expand(len(state))
            next_time = next_time.expand(len(state))
            path_t, path_s = algo.schedule(time), algo.schedule(next_time)
            probabilities = predict(state, time, path_t).float().exp()
            mean = algo.representation.posterior_mean(probabilities)
            noise = (
                torch.zeros_like(state)
                if index == self.steps - 1 and not self.final_transition_noise
                else torch.randn_like(state)
            )
            state = algo.process.reverse_posterior(state, mean, path_t, path_s, noise)
            state = clamp_prompt(state, path_s)
            return state

        state = FixedGridLoop()(state, times, update)
        final_time = times[-1].expand(len(state))
        final = predict(state, final_time, algo.schedule(final_time))
        tokens = torch.where(prompt_mask.bool(), prompt_tokens, final.argmax(-1))
        return tokens, {"nfe": nfe, "nfe_per_sample": [nfe] * len(tokens)}

"""Copied Algorithm 2 loop adapted to explicit core processes and local sessions."""

import math
import torch
from .loop import FixedGridLoop
from core.modifiers.self_conditioning.session import prediction_session
from core.time_sampling.simplex_grid import SimplexInferenceGrid
from .simplex_utils import guided_log_probs, confidence_commit


class SimplexBridgeSampler:
    def __init__(
        self,
        steps=180,
        churn=1.0,
        time_grid="linear",
        guidance_scale=None,
        confidence_power=None,
    ):
        if steps <= 0:
            raise ValueError("sampling steps must be positive")
        if guidance_scale is not None and (
            not math.isfinite(guidance_scale) or guidance_scale < 0
        ):
            raise ValueError("guidance_scale must be nonnegative")
        if confidence_power is not None and (
            not math.isfinite(confidence_power) or confidence_power <= 0
        ):
            raise ValueError("confidence_power must be positive")
        if not 0 <= churn <= 1:
            raise ValueError("churn must be in [0,1]")
        self.steps, self.churn, self.time_grid = steps, churn, time_grid
        self.guidance, self.confidence_power = guidance_scale, confidence_power

    @torch.no_grad()
    def sample(self, algo, prompt_tokens, prompt_mask):
        conditional = prediction_session(algo)
        unconditional = prediction_session(algo)
        batch = len(prompt_tokens)
        device = prompt_tokens.device
        times = (
            SimplexInferenceGrid(self.time_grid, algo.time_sampler)
            .build(self.steps, device)
            .double()
        )
        state = algo.process.prior(
            batch, algo.length, algo.schedule(times[0].expand(batch))
        )
        prompt = algo.process.encode(prompt_tokens)
        mask = prompt_mask.bool()[..., None]
        if self.guidance is not None:
            if not prompt_mask.any():
                raise ValueError(
                    "classifier-free guidance requires a conditioning prompt"
                )
            if algo.condition_dropout <= 0:
                raise ValueError(
                    "classifier-free guidance requires condition-dropout training"
                )
        tokens = None

        def update(state, time, next_time, index):
            nonlocal tokens
            state = torch.where(mask, prompt, state)
            t = time.expand(batch)
            prediction = conditional.sampling_prediction(state, t.float())
            if self.guidance is not None:
                unconditioned = unconditional.sampling_prediction(
                    state.masked_fill(mask, -torch.inf), t.float()
                )
                prediction = guided_log_probs(unconditioned, prediction, self.guidance)
            tokens = torch.distributions.Categorical(logits=prediction).sample()
            if index == self.steps - 1:
                if self.confidence_power is not None:
                    tokens = prediction.argmax(-1)
                return state
            s = next_time.expand_as(t)
            state = algo.process.reverse(
                state, tokens, algo.schedule(s), algo.schedule(t), self.churn
            )
            state = confidence_commit(state, prediction, s, mask, self.confidence_power)
            return state

        state = FixedGridLoop()(state, times, update)
        tokens = torch.where(prompt_mask.bool(), prompt_tokens, tokens)
        nfe = self.steps * (2 if self.guidance is not None else 1)
        return tokens, {"nfe": nfe, "nfe_per_sample": [nfe] * batch}

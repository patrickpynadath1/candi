"""Discrete-time absorbing diffusion training and predictor orchestration."""

import torch
from .discrete import DiscreteDiffusion, reduce_token_loss
from core.noise_schedules.log_linear import LogLinear
from core.parameterizations.posterior import PosteriorParameterization
from core.parameterizations.constraints import UnconstrainedPosterior, TokenExclusion
from core.objectives.d3pm import D3PMAbsorbingObjective
from core.processes.absorbing import AbsorbingProcess
from core.time_sampling.discrete import DiscreteTimeSampler


class D3PMAbsorb(DiscreteDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.schedule, LogLinear):
            raise TypeError("D3PM objective requires a log-linear keep schedule")
        if not isinstance(self.predictor, PosteriorParameterization) or not isinstance(
            self.predictor.constraints, (UnconstrainedPosterior, TokenExclusion)
        ):
            raise TypeError(
                "D3PM requires token probabilities without observed-token clamping"
            )
        if not isinstance(self.process, AbsorbingProcess) or not isinstance(
            self.objective, D3PMAbsorbingObjective
        ):
            raise TypeError("D3PM requires absorbing process and D3PM objective")
        if (
            not isinstance(self.time_sampler, DiscreteTimeSampler)
            or self.time_sampler.steps != self.objective.steps
        ):
            raise ValueError("D3PM time sampler and objective step counts must match")
        if self.objective.mask_index != self.process.mask_index:
            raise ValueError("D3PM mask indices must match")

    def training_prediction(self, state, keep):
        return self.predictor.training_posterior(self.model_logits(state, keep), state)

    def sampling_prediction(self, state, keep):
        return self.predictor.sampling_posterior(self.model_logits(state, keep), state)

    def greedy_prediction(self, state, keep):
        return self.predictor.greedy_posterior(self.model_logits(state, keep), state)

    def loss(
        self,
        tokens,
        loss_mask,
        corruption_mask=None,
        *,
        step=0,
        training=False,
        **time_context,
    ):
        corruption_mask = loss_mask if corruption_mask is None else corruption_mask
        time = self.time_sampler.sample(len(tokens), tokens.device, **time_context)
        keep, _ = self.schedule(time)
        state = self.process.corrupt(tokens, keep, corruption_mask)
        prediction = self.training_prediction(state, keep)
        clean_keep, _ = self.schedule(torch.zeros_like(time))
        reconstruction = self.training_prediction(tokens, clean_keep)
        loss = self.objective(prediction, state, tokens, keep, reconstruction)
        return reduce_token_loss(loss, loss_mask)

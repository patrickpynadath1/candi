"""Frozen copy of MDLM before it moved onto the view path (unify-2-hybrid-modular)."""

from core.algorithms.adaptive import validate_adaptive_feedback
from core.algorithms.training_loss import finish_training_loss

import torch
from core.algorithms.posterior import PosteriorDiffusion
from core.objectives.cross_entropy import CrossEntropy
from core.objectives.weighting import ObjectiveContext


class MDLM(PosteriorDiffusion):
    def __init__(
        self,
        model,
        predictor,
        process,
        schedule,
        time_sampler,
        objective,
        length=180,
        time_conditioning=False,
        prediction=None,
    ):
        super().__init__(prediction)
        self.model, self.predictor, self.process = model, predictor, process
        self.schedule, self.time_sampler, self.objective = (
            schedule,
            time_sampler,
            objective,
        )
        self.length, self.time_conditioning = length, time_conditioning
        validate_adaptive_feedback(schedule, predictor)
        self.validate_prediction()

    def model_inputs(self, state, keep):
        condition = -keep.clamp_min(torch.finfo(keep.dtype).tiny).log() if self.time_conditioning else torch.zeros_like(keep)
        return state, condition

    def loss(
        self,
        tokens,
        loss_mask,
        corruption_mask=None,
        *,
        step=0,
        training=False,
        time_sampler=None,
        objective=None,
        pure_noise=False,
        **time_context,
    ):
        time_sampler = self.time_sampler if time_sampler is None else time_sampler
        objective = self.objective if objective is None else objective
        if corruption_mask is None:
            corruption_mask = loss_mask
        times = time_sampler.sample(len(tokens), tokens.device, **time_context)
        keep, derivative = self.schedule(times)
        if pure_noise:
            if not isinstance(objective, CrossEntropy):
                raise ValueError("pure-noise discrete training requires a CE objective")
            keep = torch.zeros_like(keep)
        noisy = self.process.corrupt(tokens, keep, corruption_mask)
        prediction = self.training_prediction(
            noisy, keep
        )  # Shaping cannot enter this path.
        if isinstance(objective, CrossEntropy):
            per_token = objective(
                prediction, tokens,
                context=ObjectiveContext(time=times, alpha=keep, sigma=-keep.clamp_min(torch.finfo(keep.dtype).tiny).log()),
            )
        else:
            per_token = objective(prediction, tokens, keep, derivative)
        return finish_training_loss(
            per_token,
            loss_mask,
            tokens=tokens,
            prediction=prediction,
            step=step,
            training=training,
            time_sampler=time_sampler,
            sampler_time=times,
            schedule=self.schedule,
            schedule_time=times,
        )

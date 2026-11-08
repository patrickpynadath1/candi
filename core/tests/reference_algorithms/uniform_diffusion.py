"""Frozen copy of UniformDiffusion (and its DiscreteDiffusion base) before it
moved onto the view path (unify-4-foundation)."""


from core.objectives.cross_entropy import CrossEntropy
from core.objectives.weighting import ObjectiveContext

from core.algorithms.adaptive import validate_adaptive_feedback
from core.algorithms.training_loss import finish_training_loss

import torch
from torch import nn
from core.parameterizations.plug_in import PlugInParameterization
from core.processes.uniform import UniformProcess
from core.objectives.uniform import UniformDiffusionObjective
from core.modifiers.self_conditioning.prediction import PredictionCoordinator
from core.modifiers.self_conditioning.sources import posterior_mean


class DiscreteDiffusion(nn.Module):
    def __init__(
        self,
        model,
        predictor,
        process,
        schedule,
        time_sampler,
        objective,
        length=180,
        prediction=None,
    ):
        super().__init__()
        self.model, self.predictor, self.process = model, predictor, process
        self.schedule, self.time_sampler, self.objective = (
            schedule,
            time_sampler,
            objective,
        )
        self.length = length
        self.prediction = prediction
        if (
            prediction is None
            and getattr(model, "conditioning_adapter", None) is not None
        ):
            raise ValueError(
                "self-conditioning prediction is not yet integrated for this algorithm"
            )
        validate_adaptive_feedback(schedule, predictor)
        if getattr(schedule, "output_kind", None) != "keep_rate":
            raise TypeError("discrete diffusion requires a keep-rate noise schedule")
        if not model.adaLN:
            raise ValueError(
                "this discrete method requires a time-conditioned model (adaLN=true)"
            )

    def model_logits(self, state, keep):
        return self.model(state, -keep.clamp_min(torch.finfo(keep.dtype).tiny).log())

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
        state = self.process.corrupt(tokens, keep, corruption_mask)
        prediction = self.training_prediction(state, keep)
        if isinstance(objective, CrossEntropy):
            per_token = objective(
                prediction, tokens,
                context=ObjectiveContext(time=times, alpha=keep, sigma=-keep.clamp_min(1e-30).log()),
            )
        else:
            per_token = objective(
                prediction, state, tokens, keep[:, None], derivative[:, None]
            )
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

    @torch.no_grad()
    def generate(self, sampler, prompt_tokens, prompt_mask):
        was_training = self.training
        self.eval()
        try:
            return sampler.sample(self, prompt_tokens, prompt_mask)
        finally:
            self.train(was_training)


class UniformDiffusion(DiscreteDiffusion):
    def __init__(self, *args, prediction=None, time_conditioning=True, **kwargs):
        super().__init__(
            *args,
            prediction=prediction
            if prediction is not None
            else PredictionCoordinator(),
            **kwargs,
        )
        self.time_conditioning = time_conditioning
        self.prediction.validate(
            self.model, self.predictor, training_pathway="training_plugin"
        )
        if not isinstance(self.predictor, PlugInParameterization):
            raise TypeError("UniformDiffusion requires PlugInParameterization")
        if not isinstance(self.process, UniformProcess) or not isinstance(
            self.objective, UniformDiffusionObjective
        ):
            raise TypeError("UniformDiffusion requires a uniform process and objective")
        if (
            self.process.vocab_size != self.model.output_layer.linear.out_features
            or self.objective.vocab_size != self.process.vocab_size
        ):
            raise ValueError(
                "uniform model, process and objective vocabulary sizes must agree"
            )

    def prediction_mean(self, probabilities):
        return posterior_mean(probabilities, self.model.vocab_embed.embedding)

    def _prediction(self, pathway, state, keep, trajectory=None):
        return self.prediction.predict(
            self.model,
            self.predictor,
            (state, -keep.clamp_min(torch.finfo(keep.dtype).tiny).log() if self.time_conditioning else torch.zeros_like(keep)),
            state,
            pathway,
            mean=self.prediction_mean,
            trajectory=trajectory,
            training_pathway="training_plugin",
        )

    def training_prediction(self, state, keep):
        return self._prediction("training_plugin", state, keep)

    def sampling_prediction(self, state, keep, trajectory=None):
        return self._prediction("sampling_plugin", state, keep, trajectory)

    def greedy_prediction(self, state, keep, trajectory=None):
        return self._prediction("greedy_plugin", state, keep, trajectory)

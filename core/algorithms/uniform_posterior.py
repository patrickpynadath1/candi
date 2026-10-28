"""Uniform corruption trained as clean-token posterior prediction with CE."""

import torch

from .posterior import PosteriorDiffusion
from .discrete import reduce_token_loss
from .adaptive import validate_adaptive_feedback, observe_training_loss
from core.objectives.cross_entropy import CrossEntropy
from core.parameterizations.posterior import PosteriorParameterization
from core.parameterizations.constraints import UnconstrainedPosterior
from core.processes.uniform_posterior import PosteriorUniformProcess


class PosteriorUniformDiffusion(PosteriorDiffusion):
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
        super().__init__(prediction)
        self.model, self.predictor, self.process = model, predictor, process
        self.schedule, self.time_sampler, self.objective = (
            schedule,
            time_sampler,
            objective,
        )
        self.length = length
        if (
            not isinstance(predictor, PosteriorParameterization)
            or not isinstance(predictor.constraints, UnconstrainedPosterior)
            or predictor.mask_index is not None
        ):
            raise TypeError(
                "posterior uniform diffusion requires an unconstrained posterior; noisy tokens are not revealed clean tokens"
            )
        if not isinstance(process, PosteriorUniformProcess):
            raise TypeError(
                "posterior uniform diffusion requires PosteriorUniformProcess"
            )
        if not isinstance(objective, CrossEntropy) or objective.requires_context:
            raise TypeError(
                "posterior uniform diffusion requires cross entropy without Gaussian path weighting"
            )
        if getattr(schedule, "output_kind", None) != "keep_rate":
            raise TypeError("posterior uniform diffusion requires a keep-rate schedule")
        if not model.adaLN:
            raise ValueError(
                "posterior uniform diffusion requires time conditioning (adaLN=true)"
            )
        if process.vocab_size != model.output_layer.linear.out_features:
            raise ValueError("model and process vocabulary sizes must agree")
        validate_adaptive_feedback(schedule, predictor)
        self.validate_prediction()

    def model_inputs(self, state, keep):
        return state, -keep.clamp_min(torch.finfo(keep.dtype).tiny).log()

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
        corruption_mask = loss_mask if corruption_mask is None else corruption_mask
        time_sampler = self.time_sampler if time_sampler is None else time_sampler
        objective = self.objective if objective is None else objective
        times = time_sampler.sample(len(tokens), tokens.device, **time_context)
        keep, _ = self.schedule(times)
        if pure_noise:
            keep = torch.zeros_like(keep)
        state = self.process.corrupt(tokens, keep, corruption_mask)
        prediction = self.training_prediction(state, keep)
        per_token = objective(prediction, tokens)
        observe_training_loss(
            self.schedule,
            times,
            prediction,
            tokens,
            per_token,
            loss_mask,
            step=step,
            training=training,
        )
        return reduce_token_loss(per_token, loss_mask)

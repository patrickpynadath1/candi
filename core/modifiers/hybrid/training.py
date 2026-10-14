"""Parallel-view hybrid diffusion with one shared clean-token posterior."""

import torch
from .views import point, clamp

from core.algorithms.posterior import PosteriorDiffusion
from core.algorithms.adaptive import validate_adaptive_feedback, observe_training_loss
from core.objectives.weighting import ObjectiveContext


class HybridDiffusion(PosteriorDiffusion):
    supports_sequence_layout = True

    def __init__(
        self,
        model,
        predictor,
        time_sampler,
        objective,
        length=180,
        prediction=None,
        synchronize_committed=False,
    ):
        super().__init__(prediction)
        if not getattr(model, "supports_fused_streams", False):
            raise TypeError("HybridDiffusion requires a fused-stream hybrid model")
        if not callable(getattr(predictor, "training_posterior", None)):
            raise TypeError(
                "HybridDiffusion requires a clean-token posterior parameterization"
            )
        for view in model.views:
            validate_adaptive_feedback(view.schedule, predictor)
            if getattr(view.process, "prediction_kind", "posterior") != "posterior":
                raise TypeError(
                    "hybrid views must consume clean-token posteriors; "
                    "for uniform noise select process=uniform_posterior "
                    "(PosteriorUniformProcess), not the plug-in uniform process"
                )
        self.model, self.predictor = model, predictor
        self.time_sampler, self.objective = time_sampler, objective
        self.length = length
        self.synchronize_committed = bool(synchronize_committed)
        self.validate_prediction()

    @property
    def views(self):
        return self.model.views

    def points(
        self, global_time, *, training=False, corruption_mask=None, sequence_layout=None
    ):
        return tuple(
            point(
                view,
                global_time,
                training=training,
                corruption_mask=corruption_mask,
                sequence_layout=sequence_layout,
            )
            for view in self.views
        )

    def model_inputs(self, states, points, global_time):
        return states, points, global_time

    def prediction_mean(self, probabilities):
        # Exactly one fused vector is fed back, regardless of view count.
        return self.model.clean_embedding(probabilities)

    def _observed(self, states):
        for state in states:
            if state.ndim == 2 and state.dtype in (torch.int32, torch.int64):
                return state
        return torch.zeros(
            states[0].shape[:2], dtype=torch.long, device=states[0].device
        )

    def _prediction(self, pathway, states, points, global_time, trajectory=None):
        result = self.prediction.predict(
            self.model,
            self.predictor,
            self.model_inputs(states, points, global_time),
            self._observed(states),
            pathway,
            mean=self.prediction_mean,
            logits_transform=self.prediction_logits,
            trajectory=trajectory,
        )
        for view, state in zip(self.views, states):
            observed_tokens = getattr(view.process, "observed_tokens", None)
            if observed_tokens is not None:
                observed, committed = observed_tokens(state)
                fixed = torch.full_like(result, -torch.inf)
                fixed.scatter_(
                    -1, observed.clamp_max(result.shape[-1] - 1)[..., None], 0
                )
                result = torch.where(committed[..., None], fixed, result)
        return result

    def loss(
        self,
        tokens,
        loss_mask,
        corruption_mask=None,
        *,
        step=0,
        training=False,
        sequence_layout=None,
        **time_context,
    ):
        if corruption_mask is None:
            corruption_mask = loss_mask
        time = self.time_sampler.sample(len(tokens), tokens.device, **time_context)
        points = self.points(
            time,
            training=True,
            corruption_mask=corruption_mask,
            sequence_layout=sequence_layout,
        )
        states = tuple(
            view.process.corrupt_state(
                view.representation.encode(tokens), local.coefficients, corruption_mask
            )
            for view, local in zip(self.views, points)
        )
        if self.synchronize_committed:
            states = self._synchronize_committed(states)
        prediction = self.training_prediction(states, points, time)
        per_token = self.objective(
            prediction,
            tokens,
            **(
                {"context": ObjectiveContext(time=time)}
                if getattr(self.objective, "requires_context", False)
                else {}
            ),
        )
        weights = loss_mask.to(per_token)
        count = weights.sum()
        if count <= 0:
            raise ValueError("loss mask must include at least one target")
        if training:
            observe_time = getattr(self.time_sampler, "observe", None)
            if observe_time is not None:
                observe_time(time, per_token, loss_mask, step)
            for view, point in zip(self.views, points):
                observe_training_loss(
                    view.schedule,
                    point.time,
                    prediction,
                    tokens,
                    per_token,
                    loss_mask,
                    step=step,
                    training=True,
                )
        total = (per_token * weights).sum()
        return {
            "loss": total / count,
            "loss_sum": total.detach(),
            "num_tokens": count.detach(),
        }

    def _synchronize_committed(self, states):
        for view, state in zip(self.views, states):
            observed_tokens = getattr(view.process, "observed_tokens", None)
            if observed_tokens is not None:
                tokens, committed = observed_tokens(state)
                states = tuple(
                    clamp(other, value, tokens, committed)
                    for other, value in zip(self.views, states)
                )
        return states

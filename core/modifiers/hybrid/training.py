"""Parallel-view hybrid diffusion with one shared clean-token posterior."""

import torch
from .views import ViewPoint, point, clamp

from core.algorithms.posterior import PosteriorDiffusion
from core.algorithms.adaptive import validate_adaptive_feedback, observe_training_loss
from core.objectives.weighting import ObjectiveContext


class HybridDiffusion(PosteriorDiffusion):
    """Views, a shared network and one clean-token prediction.

    Family recipes subclass this (see core/algorithms/mdlm.py). Class
    attributes and the ``_corrupt``/``_objective``/``_observe`` hooks are the
    extension points, so a family changes only its own module:

    - ``prediction_pathway``/``prediction_kind``: the parameterization endpoint
      prefix and the process kind it serves (posterior; plug-in for uniform).
    - ``clamp_observed``: force observed tokens (absorbing views) to one-hot.
    """

    supports_sequence_layout = True
    prediction_kind = "posterior"
    clamp_observed = True

    def __init__(
        self,
        model,
        predictor,
        time_sampler,
        objective,
        length=180,
        prediction=None,
        synchronize_committed=False,
        adaptive_time_views=None,
        views=None,
    ):
        """``views`` holds the views on the algorithm, for a network with a
        single-state input slot (e.g. a continuous network); otherwise the
        network's own views input supplies them."""
        super().__init__(prediction)
        if views is None and not getattr(model, "supports_fused_streams", False):
            raise TypeError("HybridDiffusion requires a fused-stream hybrid model")
        if views is not None and getattr(model, "views", None) is not None:
            raise ValueError("give views either to the network or to the algorithm")
        if not callable(getattr(predictor, self._pathway("training"), None)):
            raise TypeError(
                f"HybridDiffusion requires a parameterization with {self._pathway('training')}"
            )
        for view in model.views if views is None else views:
            validate_adaptive_feedback(view.schedule, predictor)
            validate_adaptive_feedback(view.time_map, predictor)
            if getattr(view.process, "prediction_kind", "posterior") != self.prediction_kind:
                raise TypeError(
                    "hybrid views must consume clean-token posteriors; "
                    "for uniform noise select process=uniform_posterior "
                    "(PosteriorUniformProcess), not the plug-in uniform process"
                    if self.prediction_kind == "posterior"
                    else f"{type(self).__name__} requires {self.prediction_kind} processes"
                )
        self.model, self.predictor = model, predictor
        if views is not None:
            # Registered after the network and predictor, where the standalone
            # classes held process, schedule and representation.
            self.own_views = torch.nn.ModuleList(views)
        self.time_sampler, self.objective = time_sampler, objective
        self.length = length
        self.synchronize_committed = bool(synchronize_committed)
        self.adaptive_time_views = tuple(adaptive_time_views or ())
        if self.adaptive_time_views:
            if any(
                type(i) is not int or i < 0 or i >= len(self.views)
                for i in self.adaptive_time_views
            ):
                raise ValueError("adaptive_time_views must contain valid view indices")
            if not hasattr(time_sampler, "uniform") or not callable(time_sampler):
                raise TypeError(
                    "view-local adaptive time requires a sampler with uniform draws "
                    "and a quantile map"
                )
        self.validate_prediction()

    @property
    def views(self):
        own = self._modules.get("own_views")
        return self.model.views if own is None else own

    def _adaptive_time(self, time):
        mapped = self.time_sampler(time)
        # A flat CDF tail can make the generalized inverse send 1 to an
        # interior time. Priors require the exact pure-noise endpoint, and
        # the clean endpoint must remain exact too. Leave interior draws alone.
        return torch.where((time == 0) | (time == 1), time, mapped)

    def points(
        self, global_time, *, training=False, corruption_mask=None, sequence_layout=None
    ):
        adaptive_time = (
            self._adaptive_time(global_time) if self.adaptive_time_views else global_time
        )
        return tuple(
            point(
                view,
                adaptive_time if index in self.adaptive_time_views else global_time,
                training=training,
                corruption_mask=corruption_mask,
                sequence_layout=sequence_layout,
            )
            for index, view in enumerate(self.views)
        )

    def model_inputs(self, states, points, global_time):
        return states, points, global_time

    def prediction_mean(self, probabilities):
        # Exactly one fused vector is fed back, regardless of view count.
        return self.model.clean_embedding(probabilities)

    def _corrupt(self, tokens, points, corruption_mask):
        """Noisy state per view (override for e.g. condition dropout)."""
        return tuple(
            view.process.corrupt_state(
                view.representation.encode(tokens), local.coefficients, corruption_mask
            )
            for view, local in zip(self.views, points)
        )

    def _objective_context(self, time, points, states):
        coefficients = tuple(local.coefficients for local in points)
        path = coefficients[0] if len(coefficients) == 1 else None
        return ObjectiveContext(
            time=time,
            alpha=getattr(path, "alpha", None),
            sigma=getattr(path, "sigma", None),
            processes=tuple(view.process for view in self.views),
            coefficients=coefficients,
            states=states,
            representations=tuple(view.representation for view in self.views),
        )

    def _objective(self, objective, prediction, tokens, time, points, states):
        """Per-token loss (override for objectives with another signature)."""
        if getattr(objective, "requires_context", False):
            return objective(
                prediction, tokens, context=self._objective_context(time, points, states)
            )
        return objective(prediction, tokens)

    def _observe(self, sampler, feedback_time, time, points, prediction, tokens, per_token, loss_mask, step):
        """Training-loss feedback to the time sampler and each view's clocks."""
        observe_time = getattr(sampler, "observe", None)
        if observe_time is not None:
            observe_time(feedback_time, per_token, loss_mask, step)
        for view, local in zip(self.views, points):
            observe_training_loss(
                view.time_map, time, prediction, tokens, per_token, loss_mask, step=step, training=True
            )
            observe_training_loss(
                view.schedule, local.time, prediction, tokens, per_token, loss_mask, step=step, training=True
            )

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
            training_pathway=self._pathway("training"),
        )
        if not self.clamp_observed:
            return result
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
        time_sampler=None,
        objective=None,
        pure_noise=False,
        **time_context,
    ):
        """``time_sampler``/``objective`` override the defaults for this call and
        ``pure_noise`` trains a one-view keep-rate recipe at keep=0 (block-size
        mixtures use these)."""
        if corruption_mask is None:
            corruption_mask = loss_mask
        default_sampler = self.time_sampler if time_sampler is None else time_sampler
        objective = self.objective if objective is None else objective
        if self.adaptive_time_views and time_sampler is not None:
            raise ValueError("per-call time samplers do not combine with adaptive views")
        # Keep the coordinator/discrete clock uniform. Only selected continuous
        # views receive the learned quantile map, in training AND generation.
        sampler = (
            default_sampler.uniform if self.adaptive_time_views else default_sampler
        )
        time = sampler.sample(len(tokens), tokens.device, **time_context)
        feedback_time = self._adaptive_time(time) if self.adaptive_time_views else time
        points = self.points(
            time,
            training=True,
            corruption_mask=corruption_mask,
            sequence_layout=sequence_layout,
        )
        if pure_noise:
            from core.objectives.cross_entropy import CrossEntropy

            if not isinstance(objective, CrossEntropy):
                raise ValueError("pure-noise training requires a CE objective")
            if len(points) != 1 or not isinstance(points[0].coefficients, tuple):
                raise ValueError("pure-noise training needs one keep-rate view")
            keep, derivative = points[0].coefficients
            points = (ViewPoint(points[0].time, (torch.zeros_like(keep), derivative)),)
        states = self._corrupt(tokens, points, corruption_mask)
        if self.synchronize_committed:
            states = self._synchronize_committed(states)
        prediction = self.training_prediction(states, points, time)
        per_token = self._objective(objective, prediction, tokens, time, points, states)
        weights = loss_mask.to(per_token)
        count = weights.sum()
        if count <= 0:
            raise ValueError("loss mask must include at least one target")
        if training:
            self._observe(
                default_sampler, feedback_time, time, points, prediction, tokens, per_token, loss_mask, step
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

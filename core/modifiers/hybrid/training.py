"""Parallel-view hybrid diffusion with one shared clean-token posterior."""

from dataclasses import fields, is_dataclass, replace

import torch
from torch.nn import functional as F
from .views import ViewPoint, point, clamp

from core.algorithms.posterior import PosteriorDiffusion
from core.algorithms.adaptive import validate_adaptive_feedback, observe_training_loss
from core.modifiers.self_conditioning.gradient_routing import detach_selected
from core.modifiers.self_conditioning.prediction import select_batch
from core.objectives.weighting import ObjectiveContext, ObjectiveResult


def detach_rows(value, rows):
    """Detach the given batch rows of a tensor, tuple or coefficient dataclass."""
    if rows is None:
        return value
    if torch.is_tensor(value):
        return detach_selected(value, rows)
    if isinstance(value, tuple):
        return tuple(detach_rows(item, rows) for item in value)
    if is_dataclass(value):
        return replace(
            value,
            **{f.name: detach_rows(getattr(value, f.name), rows) for f in fields(value)},
        )
    return value


class HybridDiffusion(PosteriorDiffusion):
    """Views, a shared network and one clean-token prediction.

    Family recipes subclass this (see core/algorithms/mdlm.py). Class
    attributes and the ``_corrupt``/``_objective``/``_observe`` hooks are the
    extension points, so a family changes only its own module:

    - ``prediction_pathway``/``prediction_kind``: the parameterization endpoint
      prefix and the process kind it serves (posterior; plug-in for uniform).
    - ``clamp_observed``: force observed tokens (absorbing views) to one-hot.

    Components extend the shared step through optional capabilities, so recipes
    (e.g. REPLAiD) need no class of their own:

    - a time sampler may return a draw object with ``.time`` (an estimator
      allocation), declare ``min_batch``, expose ``proposal_density`` (importance
      density of the local time, passed to the objective) and ``adaptation_loss``
      (a zero-valued variance surrogate added in training), and observe objective
      terms through ``observe_terms`` instead of ``observe``;
    - a schedule may expose ``training_coefficients`` (see views.point) and
      ``observe_terms``;
    - an objective may return an ``ObjectiveResult`` and declare
      ``supports_sequence_layout = False``;
    - a parameterization with ``corrected_logits`` gets the state, coefficients and
      codebook of its ``prior_view`` before every posterior pathway;
    - the prediction coordinator's ``selected_gradients``/``fixed_feedback``
      route self-conditioned rows and their bootstrap feedback;
    - ``corrupt_all`` corrupts every position, including prompts.
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
        corrupt_all=False,
    ):
        """``views`` holds the views on the algorithm, for a network with a
        single-state input slot (e.g. a continuous network); otherwise the
        network's own views input supplies them."""
        super().__init__(prediction)
        if views is None and getattr(model, "views", None) is None:
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
        self.corrupt_all = bool(corrupt_all)
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

    def _corrupt(self, tokens, points, corruption_mask, detached_rows=None):
        """Noisy state per view (override for e.g. condition dropout).

        ``detached_rows`` detaches those rows' clean encodings (denoiser-only rows).
        """
        return tuple(
            view.process.corrupt_state(
                detach_rows(view.representation.encode(tokens), detached_rows),
                local.coefficients,
                corruption_mask,
            )
            for view, local in zip(self.views, points)
        )

    def _objective_context(self, time, points, states, **entries):
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
            schedules=tuple(view.schedule for view in self.views),
            **entries,
        )

    def _objective(self, objective, prediction, tokens, time, points, states, **entries):
        """Per-token loss (override for objectives with another signature).

        ``entries`` are the optional ObjectiveContext entries (masks, time draw,
        proposal density, detached rows)."""
        if getattr(objective, "requires_context", False):
            return objective(
                prediction,
                tokens,
                context=self._objective_context(time, points, states, **entries),
            )
        return objective(prediction, tokens)

    def _observe(
        self, sampler, feedback_time, time, points, prediction, tokens, per_token, loss_mask, step,
        *, terms=None, draw=None, detached_rows=None,
    ):
        """Training-loss feedback to the time sampler and each view's clocks.

        Components with ``observe_terms`` receive the objective's named terms
        (and the time draw) instead of the total tokenwise loss."""
        observation = dict(draw=draw, loss_mask=loss_mask, detached_rows=detached_rows)
        observe_terms = getattr(sampler, "observe_terms", None)
        observe_time = getattr(sampler, "observe", None)
        if callable(observe_terms):
            if terms is None:
                raise ValueError(
                    f"{type(sampler).__name__} observes objective terms; "
                    f"{type(self.objective).__name__} reports none"
                )
            observe_terms(terms, **observation)
        elif observe_time is not None:
            observe_time(feedback_time, per_token, loss_mask, step)
        for view, local in zip(self.views, points):
            observe_training_loss(
                view.time_map, time, prediction, tokens, per_token, loss_mask, step=step, training=True
            )
            observe_training_loss(
                view.schedule, local.time, prediction, tokens, per_token, loss_mask, step=step, training=True
            )
            schedule_terms = getattr(view.schedule, "observe_terms", None)
            if callable(schedule_terms):
                schedule_terms(terms or {}, **observation)

    def _observed(self, states):
        for state in states:
            if state.ndim == 2 and state.dtype in (torch.int32, torch.int64):
                return state
        return torch.zeros(
            states[0].shape[:2], dtype=torch.long, device=states[0].device
        )

    def _corrects_logits(self):
        return callable(getattr(self.predictor, "corrected_logits", None))

    def prediction_transform(self, states, points, detached_rows=None):
        """State-dependent correction before the shared posterior pathway.

        A parameterization with ``corrected_logits`` (the Gaussian-prior
        posterior) receives its ``prior_view``'s state, coefficients and codebook;
        ``detached_rows`` detaches those rows' codebook reads."""
        if not self._corrects_logits():
            return self.prediction_logits
        index = getattr(self.predictor, "prior_view", 0)
        state, path = states[index], points[index].coefficients
        table = self.views[index].representation.codebook()
        return lambda logits: self.predictor.corrected_logits(
            logits, state, path.alpha, path.sigma, table, detached_rows
        )

    def training_prediction(self, *state, selected=None, fixed=None):
        """``selected``: denoiser-only rows chosen by the loss (also the bootstrap
        rows); ``fixed``: (tokens, corruption_mask) for clean fixed feedback."""
        options = {
            key: value
            for key, value in dict(selected=selected, fixed=fixed).items()
            if value is not None
        }
        return self._prediction(self._pathway("training"), *state, **options)

    def _clean_feedback(self, fixed, selected):
        """Bootstrap feedback: the clean token's mean at uncorrupted positions."""
        tokens, corruption_mask = fixed
        rows = tokens[selected]
        keep = ~corruption_mask[selected].bool()[..., None]
        width = getattr(self.model, "output_width", self.model.vocab_size)

        def transform(feedback):
            clean = self.prediction_mean(F.one_hot(rows, width).float()).detach()
            return torch.where(keep, clean.to(feedback), feedback)

        return transform

    def _prediction(
        self, pathway, states, points, global_time, trajectory=None, *, selected=None, fixed=None
    ):
        options = {}
        routed = selected
        if (
            pathway == self._pathway("training")
            and trajectory is None
            and self.prediction.enabled
            and (selected is not None or fixed is not None or self._corrects_logits())
        ):
            # Choose the bootstrap rows here so their own state-dependent
            # correction and feedback can be formed (the draw the coordinator
            # would make).
            if selected is None:
                selected = self.prediction.select(
                    len(states[0]), states[0].device, self.model.training
                )
            options["selected"] = selected
            options["bootstrap_logits_transform"] = self.prediction_transform(
                select_batch(states, selected), select_batch(points, selected)
            )
            if fixed is not None and self.prediction.source == "posterior_mean":
                options["bootstrap_feedback_transform"] = self._clean_feedback(fixed, selected)
        result = self.prediction.predict(
            self.model,
            self.predictor,
            self.model_inputs(states, points, global_time),
            self._observed(states),
            pathway,
            mean=self.prediction_mean,
            logits_transform=self.prediction_transform(states, points, routed),
            trajectory=trajectory,
            training_pathway=self._pathway("training"),
            **options,
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
        default_sampler = self.time_sampler if time_sampler is None else time_sampler
        objective = self.objective if objective is None else objective
        minimum = getattr(default_sampler, "min_batch", 1)
        if len(tokens) < minimum:
            # An allocating sampler needs rows for every estimator: replicate the
            # batch and report the original token count and loss sum.
            if sequence_layout is not None:
                raise ValueError("batch replication does not combine with sequence layouts")
            repeats = -(-minimum // len(tokens))
            result = self.loss(
                tokens.repeat(repeats, 1),
                loss_mask.repeat(repeats, 1),
                None if corruption_mask is None else corruption_mask.repeat(repeats, 1),
                step=step,
                training=training,
                time_sampler=time_sampler,
                objective=objective,
                pure_noise=pure_noise,
                **time_context,
            )
            result["loss_sum"] = result["loss_sum"] / repeats
            result["num_tokens"] = result["num_tokens"] / repeats
            return result
        if sequence_layout is not None and not getattr(
            objective, "supports_sequence_layout", True
        ):
            raise ValueError(
                f"{type(objective).__name__} does not support block sequence layouts"
            )
        if self.corrupt_all:
            corruption_mask = torch.ones_like(loss_mask, dtype=torch.bool)
        elif corruption_mask is None:
            corruption_mask = loss_mask
        if self.adaptive_time_views and time_sampler is not None:
            raise ValueError("per-call time samplers do not combine with adaptive views")
        # Keep the coordinator/discrete clock uniform. Only selected continuous
        # views receive the learned quantile map, in training AND generation.
        sampler = (
            default_sampler.uniform if self.adaptive_time_views else default_sampler
        )
        draw = sampler.sample(len(tokens), tokens.device, **time_context)
        time, draw = (draw, None) if torch.is_tensor(draw) else (draw.time, draw)
        feedback_time = self._adaptive_time(time) if self.adaptive_time_views else time
        proposal_density = getattr(default_sampler, "proposal_density", None)
        density = proposal_density(feedback_time) if callable(proposal_density) else None
        selected = (
            self.prediction.select(len(tokens), tokens.device, training)
            if getattr(self.prediction, "selected_gradients", "all") == "denoiser_only"
            else None
        )
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
        if selected is not None:
            # Denoiser-only rows: their schedule coefficients and clean
            # encodings do not train the schedule or representation.
            points = tuple(
                ViewPoint(local.time, detach_rows(local.coefficients, selected))
                for local in points
            )
            states = self._corrupt(tokens, points, corruption_mask, detached_rows=selected)
        else:
            states = self._corrupt(tokens, points, corruption_mask)
        if self.synchronize_committed:
            states = self._synchronize_committed(states)
        fixed = (
            (tokens, corruption_mask)
            if getattr(self.prediction, "fixed_feedback", "prediction") == "clean"
            else None
        )
        options = {
            key: value
            for key, value in dict(selected=selected, fixed=fixed).items()
            if value is not None
        }
        prediction = self.training_prediction(states, points, time, **options)
        per_token = self._objective(
            objective, prediction, tokens, time, points, states,
            loss_mask=loss_mask, corruption_mask=corruption_mask, time_draw=draw,
            time_density=density, detached_rows=selected,
        )
        result = per_token if isinstance(per_token, ObjectiveResult) else None
        if result is not None:
            per_token = result.total
        weights = loss_mask.to(per_token)
        count = weights.sum()
        if count <= 0:
            raise ValueError("loss mask must include at least one target")
        if training:
            adaptation_loss = getattr(default_sampler, "adaptation_loss", None)
            if callable(adaptation_loss):
                if result is None or result.estimate is None:
                    raise ValueError(
                        f"{type(default_sampler).__name__} needs an objective that "
                        "reports per-example estimates (ObjectiveResult.estimate)"
                    )
                per_token = per_token + adaptation_loss(result.estimate, density)[:, None] * weights
            self._observe(
                default_sampler, feedback_time, time, points, prediction, tokens, per_token,
                loss_mask, step, terms=None if result is None else result.terms, draw=draw,
                detached_rows=selected,
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

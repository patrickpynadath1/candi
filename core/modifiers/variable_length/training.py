"""Composable deletion/insertion around one or more aligned diffusion views.

The inner assembly owns representations, schedules, content updates and token
prediction. This modifier owns structure, gap supervision and alignment only.
"""

from collections.abc import Mapping
import torch
from torch import nn

from core.algorithms.adaptive import observe_training_loss
from core.modifiers.hybrid.views import point
from core.noise_schedules.insertion import PowerInsertionSchedule
from core.objectives.insertion import InsertionObjective
from core.parameterizations.posterior import PosteriorParameterization
from core.processes.deletion import DeletionProcess
from .birth import ForwardInsertion
from .model import GapHead, SequencePrediction, forward_features
from core.sequence_edits import valid_positions


class VariableLengthDiffusion(nn.Module):
    supports_variable_length = True

    def __init__(
        self,
        algorithm,
        vocab_size,
        insertion=None,
        schedule=None,
        process=None,
        objective=None,
        gap_predictor=None,
        training_epsilon=1e-4,
        time_sampling_include_aux=False,
    ):
        super().__init__()
        if not hasattr(algorithm, "views") or not callable(
            getattr(algorithm.model, "prepare_fused", None)
        ):
            raise TypeError(
                "variable length requires a DiffusionView assembly; use one view for a single process"
            )
        if not 0 < training_epsilon < 0.5:
            raise ValueError("training_epsilon must lie in (0, 0.5)")
        if getattr(algorithm.predictor, "mask_index", None) is not None:
            raise TypeError(
                "use an unconstrained token posterior; observations are enforced per view"
            )
        self.algorithm = algorithm
        if insertion is None:
            insertion = [ForwardInsertion() for _ in algorithm.views]
        if isinstance(insertion, Mapping):
            insertion = [
                insertion.get(view.name, ForwardInsertion()) for view in algorithm.views
            ]
        if len(insertion) != len(algorithm.views):
            raise ValueError("one insertion law is required per diffusion view")
        self.insertion = nn.ModuleList(insertion)
        for law, view in zip(self.insertion, self.views):
            law.validate(view)
            # An absorbing view needs the explicit coupled birth law, not the
            # generic independent forward kernel (which can reveal at birth).
            if callable(getattr(view.process, "observed_tokens", None)) and isinstance(
                law, ForwardInsertion
            ):
                raise TypeError("absorbing views require MaskInsertion")
        hidden = self.model.backbone.blocks[0].attn_qkv.in_features
        self.gap_head = GapHead(hidden, vocab_size)
        self.schedule = schedule or PowerInsertionSchedule()
        self.process = process or DeletionProcess()
        self.objective = objective or InsertionObjective()
        self.gap_predictor = gap_predictor or PosteriorParameterization()
        self.training_epsilon = training_epsilon
        self.time_sampling_include_aux = time_sampling_include_aux

    @property
    def model(self):
        return self.algorithm.model

    @property
    def views(self):
        return self.algorithm.views

    @property
    def prediction(self):
        return self.algorithm.prediction

    @property
    def length(self):
        return self.algorithm.length

    def points(self, time, *, training=False):
        return tuple(point(view, time, training=training) for view in self.views)

    def predict(self, states, points, time, valid, *, training=False, feedback=None):
        policy = self.prediction
        observed = self.algorithm._observed(states)
        nfe = 1
        if training and policy.enabled:
            selected = policy.select(len(valid), valid.device, self.model.training)
            if selected.any():
                # Keep a single aligned batch for the bootstrap. Selection gates
                # feedback, not the corruption state or structural targets.
                with (
                    torch.no_grad(),
                    torch.autocast(
                        valid.device.type,
                        enabled=torch.is_autocast_enabled(valid.device.type),
                        cache_enabled=False,
                    ),
                ):
                    bootstrap, _, _ = forward_features(
                        self.model, self.gap_head, states, points, time, valid
                    )
                    value = policy.feedback(
                        bootstrap,
                        self.algorithm.predictor,
                        observed,
                        self.algorithm.prediction_mean,
                        self.algorithm.prediction_logits,
                    )
                    feedback = torch.where(selected[:, None, None], value, 0)
                nfe += 1
        output, counts, gap_logits = forward_features(
            self.model, self.gap_head, states, points, time, valid, feedback
        )
        pathway = "training_posterior" if training else "sampling_posterior"
        token_log_probs = getattr(self.algorithm.predictor, pathway)(
            output.logits, observed
        )
        for view, state in zip(self.views, states):
            observe = getattr(view.process, "observed_tokens", None)
            if observe is not None:
                tokens, committed = observe(state)
                fixed = torch.full_like(token_log_probs, -torch.inf)
                fixed.scatter_(
                    -1, tokens.clamp_max(self.gap_head.vocab_size - 1)[..., None], 0
                )
                token_log_probs = torch.where(
                    (committed & valid)[..., None], fixed, token_log_probs
                )
        gap_log_probs = getattr(self.gap_predictor, pathway)(
            gap_logits, torch.zeros_like(counts, dtype=torch.long)
        )
        next_feedback = None
        if policy.enabled:
            next_feedback = policy.feedback(
                output,
                self.algorithm.predictor,
                observed,
                self.algorithm.prediction_mean,
                self.algorithm.prediction_logits,
            )
        return SequencePrediction(
            token_log_probs, counts, gap_log_probs, next_feedback, nfe
        )

    def loss(
        self,
        tokens,
        loss_mask,
        corruption_mask=None,
        *,
        valid_mask=None,
        step=0,
        training=False,
        time_group_size=1,
        **time_context,
    ):
        if "sequence_layout" in time_context:
            raise ValueError(
                "variable length requires explicit editable spans, not fixed-index block layouts"
            )
        if tokens.ndim != 2 or loss_mask.shape != tokens.shape:
            raise ValueError(
                "tokens and loss_mask must have matching [batch, length] shapes"
            )
        valid = (
            torch.ones_like(tokens, dtype=torch.bool)
            if valid_mask is None
            else valid_mask.bool()
        )
        editable = (
            loss_mask.bool() if corruption_mask is None else corruption_mask.bool()
        )
        if (
            valid.shape != tokens.shape
            or editable.shape != tokens.shape
            or (loss_mask.bool() & ~valid).any()
        ):
            raise ValueError("training masks must match valid tokens")
        count = loss_mask.sum()
        if count <= 0:
            raise ValueError("loss mask must include at least one target")
        if (
            not isinstance(time_group_size, int)
            or time_group_size < 1
            or len(tokens) % time_group_size
        ):
            raise ValueError("time_group_size must divide the batch size")
        times = self.algorithm.time_sampler.sample(
            len(tokens) // time_group_size, tokens.device, **time_context
        ).repeat_interleave(time_group_size)
        times = times.clamp(self.training_epsilon, 1 - self.training_epsilon)
        sample = self.process.corrupt(valid, editable, times, self.schedule)
        edit = sample.edit
        # Invalid padding token IDs never reach an embedding or target lookup.
        clean = torch.where(valid, tokens, 0)
        compact = edit.apply(clean)
        compact_editable = edit.apply(editable)
        compact_loss = edit.apply(loss_mask)
        points = self.points(times, training=True)
        states = tuple(
            law.corrupt_training(
                view, compact, local, sample.birth_times, compact_editable
            )
            for law, view, local in zip(self.insertion, self.views, points)
        )
        if self.algorithm.synchronize_committed:
            states = self.algorithm._synchronize_committed(states)
        # `training` controls feedback observations; both train and validation
        # always use the unshaped training posterior.
        prediction = self.predict(states, points, times, edit.valid, training=True)
        per_token = self.algorithm.objective(prediction.tokens, compact)
        token_sum = torch.where(compact_loss.bool(), per_token * compact_loss, 0).sum()
        # Gaps containing removed targets, and gaps adjacent to editable tokens,
        # are eligible. Fully protected interiors never learn to insert.
        allowed = torch.zeros_like(sample.gap_counts, dtype=torch.bool)
        slots = sample.gap_for_token.clamp_max(allowed.shape[1] - 1)
        hits = torch.zeros_like(sample.gap_counts)
        hits.scatter_add_(1, slots, (editable & valid).long())
        hits.scatter_add_(
            1,
            (slots + 1).clamp_max(allowed.shape[1] - 1),
            (editable & valid & ~sample.deleted).long(),
        )
        allowed = (hits > 0) & valid_positions(edit.lengths + 1, allowed.shape[1])
        insertion_sum, counts, content = self.objective(
            prediction.expected_counts,
            prediction.gap_tokens,
            sample,
            clean,
            allowed,
            self.schedule.rate(times),
            **({"reduction": "none"} if self.time_sampling_include_aux else {}),
        )
        if self.time_sampling_include_aux:
            insertion_per_example = insertion_sum
            insertion_sum, counts, content = (
                insertion_sum.sum(),
                counts.sum(),
                content.sum(),
            )
        if training:
            for view, local in zip(self.views, points):
                observe_training_loss(
                    view.schedule,
                    local.time,
                    prediction.tokens,
                    compact,
                    per_token,
                    compact_loss,
                    step=step,
                    training=True,
                )
            observer = getattr(self.algorithm.time_sampler, "observe", None)
            if observer is not None:
                if self.time_sampling_include_aux:
                    # Use the same weighted objective and original target-token
                    # normalization as training, including fully deleted rows.
                    token_totals = torch.where(
                        compact_loss.bool(), per_token * compact_loss, 0
                    ).sum(-1)
                    target_counts = loss_mask.sum(-1)
                    combined = (token_totals + insertion_per_example).detach()
                    observer(
                        times,
                        (combined / target_counts.clamp_min(1))[:, None],
                        (target_counts > 0)[:, None],
                        step,
                    )
                else:
                    observer(times, per_token, compact_loss, step)
        total = token_sum + insertion_sum
        return {
            "loss": total / count,
            "loss_sum": total.detach(),
            "num_tokens": count.detach(),
            "content_loss": token_sum.detach() / count,
            "insertion_loss": counts.detach() / count,
            "insertion_content_loss": content.detach() / count,
        }

    @torch.no_grad()
    def generate(self, sampler, prompt_tokens, prompt_mask):
        if not getattr(sampler, "supports_variable_length", False):
            raise TypeError("variable length requires a structural sampler")
        was_training = self.training
        self.eval()
        try:
            return sampler.sample(self, prompt_tokens, prompt_mask)
        finally:
            self.train(was_training)

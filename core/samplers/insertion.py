"""Joint content updates and Poisson insertion on a shared dynamic sequence."""

import torch
from core.modifiers.hybrid.views import clamp
from core.sequence_edits import SequenceEdit, gather_positions, valid_positions


class InsertionSampler:
    supports_variable_length = True

    def __init__(
        self,
        steps=180,
        epsilon=1e-3,
        max_length=512,
        overflow="error",
        final_denoise="sample",
        pad_token_id=0,
    ):
        if not isinstance(steps, int) or steps < 1 or not 0 < epsilon < 1:
            raise ValueError("positive steps and epsilon in (0, 1) required")
        if not isinstance(max_length, int) or max_length < 1:
            raise ValueError("max_length must be positive")
        if overflow not in {"error", "truncate"} or final_denoise not in {
            "sample",
            "greedy",
        }:
            raise ValueError("unknown insertion overflow or final denoise policy")
        self.steps, self.epsilon, self.max_length = steps, epsilon, max_length
        self.overflow, self.final_denoise, self.pad_token_id = (
            overflow,
            final_denoise,
            pad_token_id,
        )

    def configure_special_tokens(self, *, pad_token_id=None, **kwargs):
        if pad_token_id is not None:
            self.pad_token_id = pad_token_id

    def _limit(self, counts, lengths, max_length=None):
        max_length = (
            self.max_length
            if max_length is None
            else min(self.max_length, int(max_length))
        )
        overflow = lengths + counts.sum(1) > max_length
        if overflow.any() and self.overflow == "error":
            raise RuntimeError(
                "insertion exceeds max_length; increase capacity or explicitly select overflow=truncate"
            )
        if overflow.any():
            counts = counts.clone()
            for row in overflow.nonzero().flatten().tolist():
                capacity = max_length - int(lengths[row])
                remaining = counts[row].clone()
                chosen = torch.zeros_like(remaining)
                for _ in range(capacity):
                    gap = torch.multinomial(remaining.float(), 1)
                    chosen[gap] += 1
                    remaining[gap] -= 1
                counts[row] = chosen
        return counts, overflow

    @torch.no_grad()
    def sample(self, algo, prompt_tokens, prompt_mask, *, editable_gaps=None):
        if not getattr(algo, "supports_variable_length", False):
            raise TypeError("InsertionSampler requires a variable-length assembly")
        if (
            prompt_tokens.ndim != 2
            or prompt_tokens.shape != prompt_mask.shape
            or len(prompt_tokens) == 0
        ):
            raise ValueError(
                "prompt tensors must have matching nonempty batch dimensions"
            )
        protected = prompt_mask.bool()
        max_length = min(
            self.max_length, int(getattr(algo, "max_length", self.max_length))
        )
        compact = SequenceEdit.deletion(protected)
        lengths = compact.lengths
        if (lengths > max_length).any():
            raise ValueError("prompt exceeds max_length")
        anchors = compact.apply(prompt_tokens)
        fixed = compact.valid
        width = anchors.shape[1]
        if editable_gaps is None:
            # Existing placeholders specify editable spans, never target lengths.
            # If the caller supplies only a compact prompt, default to appending.
            preceding = protected.long().cumsum(1)
            hits = torch.zeros(
                (len(lengths), width + 1), dtype=torch.long, device=anchors.device
            )
            hits.scatter_add_(1, preceding, (~protected).long())
            allowed = hits > 0
            no_holes = ~allowed.any(1)
            allowed[no_holes, lengths[no_holes]] = True
        else:
            allowed = editable_gaps.bool()
            if allowed.shape != (len(lengths), width + 1):
                raise ValueError(
                    "editable_gaps must describe the compacted prompt gaps"
                )
        allowed = allowed & valid_positions(lengths + 1, width + 1)
        states = tuple(view.representation.clean_state(anchors) for view in algo.views)
        # Use valid clean padding states, including simplex/spherical domains.
        feedback = None
        truncated = torch.zeros(len(lengths), dtype=torch.bool, device=anchors.device)
        inserted_total = torch.zeros_like(lengths)
        nfe = 0
        times = torch.linspace(1, self.epsilon, self.steps + 1, device=anchors.device)
        for source, destination in zip(times[:-1], times[1:]):
            t, s = source.expand(len(lengths)), destination.expand(len(lengths))
            points_t, points_s = algo.points(t), algo.points(s)
            valid = valid_positions(lengths, states[0].shape[1])
            prediction = algo.predict(states, points_t, t, valid, feedback=feedback)
            nfe += prediction.nfe
            posterior = prediction.tokens.float().exp()
            updated = (
                states
                if states[0].shape[1] == 0
                else tuple(
                    clamp(
                        view,
                        view.update(
                            view.process, view.representation, state, posterior, pt, ps
                        ),
                        anchors,
                        fixed,
                    )
                    for view, state, pt, ps in zip(
                        algo.views, states, points_t, points_s
                    )
                )
            )
            intensity = (
                prediction.expected_counts
                * algo.schedule.integrated_rate(t, s)[:, None]
            )
            if not torch.isfinite(intensity).all():
                raise RuntimeError("nonfinite insertion intensity")
            counts = torch.poisson(torch.where(allowed, intensity, 0)).long()
            counts, overflow = self._limit(counts, lengths, max_length)
            truncated |= overflow
            inserted_total += counts.sum(1)
            edit = SequenceEdit.insertion(lengths, counts)
            # Every birth gets its own random draw, including repeated births in
            # the same gap. q_s uses the DESTINATION global time, never birth age.
            birth_posterior = gather_positions(prediction.gap_tokens.exp(), edit.gap)
            new_states = []
            for view, law, old, ps in zip(
                algo.views, algo.insertion, updated, points_s
            ):
                moved = edit.apply(old)
                if edit.inserted.any():
                    births = law.initialize(view, birth_posterior, ps)
                    selector = edit.inserted.reshape(
                        *edit.inserted.shape, *([1] * (moved.ndim - 2))
                    )
                    moved = torch.where(selector, births, moved)
                # Padding must be a legal domain value even though attention and
                # losses ignore it (e.g. zero log-state is not a simplex point).
                zero_tokens = torch.zeros(
                    edit.source.shape, dtype=torch.long, device=anchors.device
                )
                padding = view.representation.clean_state(zero_tokens)
                selector = edit.valid.reshape(
                    *edit.valid.shape, *([1] * (moved.ndim - 2))
                )
                new_states.append(torch.where(selector, moved, padding))
            states = tuple(new_states)
            anchors, fixed = edit.apply(anchors), edit.apply(fixed)
            feedback = (
                None if prediction.feedback is None else edit.apply(prediction.feedback)
            )
            # A gap split by births inherits its edit permission. Protected
            # interiors remain closed, independent of shifting tensor indices.
            gap_sources = torch.where(
                edit.source >= 0, edit.source, edit.gap
            ).clamp_min(0)
            expanded = gather_positions(allowed, gap_sources)
            next_allowed = torch.zeros(
                (len(lengths), edit.source.shape[1] + 1),
                dtype=torch.bool,
                device=anchors.device,
            )
            next_allowed[:, :-1] = expanded & edit.valid
            next_allowed[
                torch.arange(len(lengths), device=anchors.device), edit.lengths
            ] = allowed[torch.arange(len(lengths), device=anchors.device), lengths]
            allowed, lengths = next_allowed, edit.lengths
            if algo.algorithm.synchronize_committed:
                states = algo.algorithm._synchronize_committed(states)
        time = times[-1].expand(len(lengths))
        valid = valid_positions(lengths, states[0].shape[1])
        prediction = algo.predict(
            states, algo.points(time), time, valid, feedback=feedback
        )
        nfe += prediction.nfe
        tokens = (
            prediction.tokens.argmax(-1)
            if self.final_denoise == "greedy" or prediction.tokens.shape[1] == 0
            else torch.distributions.Categorical(logits=prediction.tokens).sample()
        )
        tokens = torch.where(fixed, anchors, tokens)
        tokens = torch.where(valid, tokens, self.pad_token_id)
        return tokens, {
            "nfe": nfe,
            "nfe_per_sample": [nfe] * len(lengths),
            "lengths": lengths.tolist(),
            "truncated": truncated.tolist(),
            "inserted_tokens": inserted_total.tolist(),
        }

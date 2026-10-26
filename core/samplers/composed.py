"""One fixed-length sampling loop for pure and parallel-view diffusion recipes."""

import torch
from core.modifiers.hybrid.views import prior, clamp
from core.modifiers.self_conditioning.session import prediction_session
from .policies import AllPositions, TokenSelection, ReverseTransition, select_state


class ComposedSampler:
    def __init__(
        self,
        steps=180,
        epsilon=1e-5,
        *,
        tokens=None,
        positions=None,
        transition=None,
        commitment="none",
        corrector=None,
        final_denoise="greedy",
        final_prediction=None,
        synchronize_committed=False,
        require_completion=False,
    ):
        if int(steps) != steps or steps < 1 or not 0 <= epsilon < 1:
            raise ValueError("positive integer steps and epsilon in [0, 1) required")
        if commitment not in {"none", "freeze", "revise"}:
            raise ValueError("commitment must be none, freeze, or revise")
        if final_prediction is None:
            final_prediction = "greedy" if final_denoise == "greedy" else "sampling"
        if final_denoise not in {"greedy", "sample"} or final_prediction not in {
            "greedy",
            "sampling",
        }:
            raise ValueError("invalid terminal token selection or prediction pathway")
        self.steps, self.epsilon = int(steps), float(epsilon)
        self.tokens = tokens if tokens is not None else TokenSelection("posterior")
        self.positions = positions if positions is not None else AllPositions()
        self.transition = transition if transition is not None else ReverseTransition()
        self.corrector = corrector
        if corrector is not None and commitment != "none":
            raise ValueError("correctors currently require commitment=none")
        self.commitment = commitment
        self.final_denoise, self.final_prediction = final_denoise, final_prediction
        self.synchronize_committed = bool(synchronize_committed)
        self.require_completion = bool(require_completion)
        if self.tokens.mode == "posterior" and (
            self.transition.requires_tokens or commitment != "none"
        ):
            raise ValueError(
                "forward corruption and commitments require sample or greedy token selection"
            )

    def steps_for_nfe(self, nfe):
        """Largest complete predictor/corrector trajectory within a call budget."""
        if isinstance(nfe, bool) or int(nfe) != nfe:
            raise ValueError("NFE budget must be an integer")
        calls_per_step = 1 + (self.corrector.steps if self.corrector is not None else 0)
        steps = (int(nfe) - 1) // calls_per_step  # Reserve terminal decoding.
        if steps < 1:
            raise ValueError(
                "NFE budget cannot fit a predictor/corrector step and decoding"
            )
        return steps

    @torch.no_grad()
    def sample(self, algo, prompt_tokens, prompt_mask):
        if getattr(algo, "supports_variable_length", False):
            raise TypeError(
                "fixed-length sampling requires a separate policy for length edits"
            )
        if not hasattr(algo, "views") or not callable(getattr(algo, "points", None)):
            raise TypeError(
                "sampling requires the single-view/parallel-view algorithm interface"
            )
        self.transition.validate(algo.views, self.commitment)
        if self.corrector is not None:
            self.corrector.validate(algo.views)
        algo = prediction_session(algo)
        batch = len(prompt_tokens)
        prompt_mask = prompt_mask.bool()
        generated = ~prompt_mask
        times = torch.linspace(
            1, self.epsilon, self.steps + 1, device=prompt_tokens.device
        )
        points = algo.points(times[0].expand(batch), corruption_mask=generated)
        states = tuple(
            prior(v, prompt_tokens, p, prompt_mask) for v, p in zip(algo.views, points)
        )
        committed = prompt_mask.clone()
        complete = ~generated.any(-1)
        output = prompt_tokens.clone()
        per_sample = torch.zeros(batch, dtype=torch.long, device=prompt_tokens.device)
        counts = []
        nfe = 0
        for index in range(self.steps):
            active = (
                generated & ~committed if self.commitment == "freeze" else generated
            )
            if self.commitment != "none":
                active = active & ~complete[:, None]
                if not active.any():
                    break
            time, next_time = times[index].expand(batch), times[index + 1].expand(batch)
            source = algo.points(time, corruption_mask=generated)
            destination = algo.points(next_time, corruption_mask=generated)
            logits = algo.sampling_prediction(states, source, time)
            probabilities = logits.float().exp()
            selected = self.positions(probabilities, active)
            clean = self.tokens(logits, probabilities)
            next_output = (
                output if clean is None else torch.where(selected, clean, output)
            )
            next_committed = (
                committed | selected
                if self.commitment == "freeze"
                else prompt_mask | selected
                if self.commitment == "revise"
                else prompt_mask
            )
            released = committed & generated & ~selected & ~complete[:, None]
            updated = []
            for view, state, pt, ps in zip(algo.views, states, source, destination):
                value = self.transition(
                    view,
                    state,
                    probabilities,
                    clean,
                    pt,
                    ps,
                    final_step=index == self.steps - 1,
                    generated=generated,
                    prompt_tokens=prompt_tokens,
                    released=released,
                    previous_tokens=output,
                )
                if self.commitment == "none":
                    value = select_state(selected, value, state)
                value = clamp(view, value, next_output, next_committed)
                if self.commitment != "none":
                    value = select_state(complete, state, value)
                updated.append(value)
            states = tuple(updated)
            if self.synchronize_committed:
                states = algo._synchronize_committed(states)
            nfe += 1
            if self.corrector is not None:
                for _ in range(self.corrector.steps):
                    posterior = (
                        algo.sampling_prediction(states, destination, next_time)
                        .float()
                        .exp()
                    )
                    states = tuple(
                        clamp(
                            view,
                            select_state(
                                selected,
                                self.corrector(view, state, posterior, point),
                                state,
                            ),
                            prompt_tokens,
                            prompt_mask,
                        )
                        for view, state, point in zip(algo.views, states, destination)
                    )
                    nfe += 1
            if self.commitment != "none":
                output = torch.where(complete[:, None], output, next_output)
                committed = next_committed
                counts.append(selected.sum(-1).cpu().tolist())
                newly_complete = ~complete & committed.all(-1)
                per_sample[newly_complete] = nfe
                complete |= newly_complete
        if self.commitment == "none" or not complete.all():
            if self.require_completion:
                raise RuntimeError("masked hybrid EB exhausted steps before completion")
            time = times[-1].expand(batch)
            points = algo.points(time, corruption_mask=generated)
            prediction = (
                algo.greedy_prediction(states, points, time)
                if self.final_prediction == "greedy"
                else algo.sampling_prediction(states, points, time)
            )
            final_tokens = (
                prediction.argmax(-1)
                if self.final_denoise == "greedy"
                else torch.distributions.Categorical(logits=prediction).sample()
            )
            nfe += 1
            if self.commitment == "none":
                output = final_tokens
                per_sample[:] = nfe
            else:
                output = torch.where(
                    complete[:, None] | committed, output, final_tokens
                )
                per_sample[~complete] = nfe
        output = torch.where(prompt_mask, prompt_tokens, output)
        info = dict(nfe=nfe, nfe_per_sample=per_sample.cpu().tolist())
        if self.commitment != "none":
            info.update(
                accepted_per_step=counts, completed_per_sample=complete.cpu().tolist()
            )
        return output, info

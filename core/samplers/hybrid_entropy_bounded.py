"""Entropy-bounded acceptance over a parallel hybrid trajectory."""

import math
import torch

from core.hybrid.branches import BranchPoint
from core.modifiers.self_conditioning.session import prediction_session
from .entropy_bounded import select_prefix


class HybridEntropyBoundedSampler:
    def __init__(self, entropy_bound=0.1, steps=180,
                 greedy_clean_estimate=True):
        if not math.isfinite(entropy_bound) or entropy_bound < 0 or steps < 1:
            raise ValueError("finite nonnegative entropy bound and positive steps required")
        self.bound, self.steps = float(entropy_bound), int(steps)
        self.greedy_clean_estimate = bool(greedy_clean_estimate)

    @torch.no_grad()
    def sample(self, algo, prompt_tokens, prompt_mask):
        algo = prediction_session(algo)
        batch, length = prompt_tokens.shape
        # Historical CANDI solver grid deliberately stops short of both endpoints.
        times = torch.linspace(.999, 1e-5, self.steps + 1, device=prompt_tokens.device)
        source = algo.points(times[0].expand(batch))
        source = tuple(
            BranchPoint(times[0].expand(batch), branch.schedule(times[0].expand(batch)))
            if index > 0 else point
            for index, (branch, point) in enumerate(zip(algo.branches, source))
        )
        states = tuple(
            branch.prior(prompt_tokens, point, prompt_mask)
            for branch, point in zip(algo.branches, source)
        )
        accepted = prompt_mask.bool().clone()
        output = prompt_tokens.clone()
        per_sample = torch.zeros(batch, dtype=torch.long, device=output.device)
        accepted_per_step = []
        nfe = 0
        for index in range(self.steps):
            active = ~accepted
            if not active.any():
                break
            time = times[index].expand(batch)
            next_time = times[index + 1].expand(batch)
            source, destination = algo.points(time), algo.points(next_time)
            prediction_points = list(source)
            # CANDI predicts with the source discrete clock and destination
            # Gaussian sigma. This asymmetry is part of its saved baseline.
            for branch_index, state in enumerate(states):
                if state.ndim == 3:
                    prediction_points[branch_index] = destination[branch_index]
            discrete_keep = source[0].coefficients
            if isinstance(discrete_keep, tuple):
                discrete_keep = discrete_keep[0]
            discrete_noise = 1 - discrete_keep
            prediction = algo.sampling_prediction(
                states, tuple(prediction_points), discrete_noise
            )
            probabilities = prediction.double().exp()
            entropy_probabilities = probabilities.float()
            entropy = -torch.special.xlogy(
                entropy_probabilities, entropy_probabilities
            ).sum(-1)
            selected = select_prefix(entropy, active, self.bound)
            clean = (
                probabilities.argmax(-1) if self.greedy_clean_estimate
                else torch.distributions.Categorical(probabilities).sample()
            )
            output = torch.where(selected, clean, output)
            next_states = []
            for branch, state, point_t, point_s in zip(
                algo.branches, states, source, destination
            ):
                if state.ndim == 2:
                    continued = state
                else:
                    continued = branch.update(
                        state, probabilities, point_t, point_s,
                        noise=torch.zeros_like(state),
                    )
                newly_clean = accepted | selected
                next_states.append(branch.clamp_prompt(
                    continued, output, newly_clean, point_s
                ))
            states = tuple(next_states)
            accepted |= selected
            nfe += 1
            counts = selected.sum(-1)
            accepted_per_step.append(counts.cpu().tolist())
            complete = accepted.all(-1)
            per_sample[(per_sample == 0) & complete] = nfe
        if not accepted.all():
            raise RuntimeError(
                "hybrid entropy-bounded sampling exhausted steps before completion"
            )
        return output, {
            "nfe": nfe,
            "nfe_per_sample": per_sample.cpu().tolist(),
            "accepted_per_step": accepted_per_step,
            "entropy_bound": self.bound,
            "greedy_clean_estimate": self.greedy_clean_estimate,
        }

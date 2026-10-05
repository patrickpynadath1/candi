"""Step-local entropy-bounded commitment with revisable uniform states.

By default only prompts are fixed. Every generated position participates at every step,
including positions committed on earlier steps. A sequence finishes when all
its generated positions are selected in the same step. Set freeze=True for
cumulative acceptance with permanently fixed committed tokens.
"""

import math
import torch
from .uniform_ancestral import initialize_uniform
from .entropy_bounded import select_prefix
from core.modifiers.self_conditioning.session import prediction_session


class UniformEntropyBoundedSampler:
    def __init__(self, entropy_bound=0.0, steps=1024, epsilon=1e-3, use_float64=True, freeze=False):
        if (
            not math.isfinite(entropy_bound)
            or entropy_bound < 0
            or not isinstance(steps, int)
            or steps <= 0
            or not 0 < epsilon < 1
        ):
            raise ValueError("invalid entropy bound or sampling grid")
        self.freeze = bool(freeze)
        self.bound, self.steps, self.epsilon, self.use_float64 = (
            entropy_bound,
            steps,
            epsilon,
            use_float64,
        )

    @torch.no_grad()
    def sample(self, algo, prompt_tokens, prompt_mask):
        algo = prediction_session(algo)
        state = initialize_uniform(algo, prompt_tokens, prompt_mask)
        generated = ~prompt_mask.bool()
        complete = ~generated.any(-1)
        committed = prompt_mask.bool().clone()
        counts = []
        per_sample = torch.zeros(len(state), dtype=torch.long, device=state.device)
        times = torch.linspace(1, self.epsilon, self.steps + 1, device=state.device)
        for time, next_time in zip(times[:-1], times[1:]):
            active = generated & ~complete[:, None]
            if self.freeze:
                active &= ~committed
            if not active.any():
                break
            keep, _ = algo.schedule(time.expand(len(state)))
            next_keep, _ = algo.schedule(next_time.expand(len(state)))
            prediction = algo.sampling_prediction(state, keep)
            # Match the original precision: full probabilities in double for
            # reverse transitions, entropy and multinomial commitment in float.
            prediction = prediction.double() if self.use_float64 else prediction.float()
            probabilities = algo.predictor.probabilities(
                prediction, apply_commitment=True
            )
            # Posterior bridges use the full posterior; commitment only affects
            # accepted tokens. Preserve the historical plug-in update policy.
            transition_probabilities = (
                algo.predictor.probabilities(prediction)
                if algo.process.prediction_kind == "posterior"
                else probabilities
            )
            rejected = algo.process.reverse(
                state, transition_probabilities, keep, next_keep
            )
            selected = select_prefix(
                algo.predictor.entropy(prediction), active, self.bound
            )
            clean = algo.process.commit(
                probabilities.float(),
                deterministic=algo.predictor.commitment == "greedy",
            )
            if self.freeze:
                rejected = torch.where(committed, state, rejected)
            updated = torch.where(selected, clean, rejected)
            state = torch.where(complete[:, None], state, updated)
            state = torch.where(prompt_mask.bool(), prompt_tokens, state)
            counts.append(selected.sum(-1).cpu().tolist())
            committed = (committed | selected) if self.freeze else (selected | ~generated)
            newly_complete = ~complete & committed.all(-1)
            per_sample[newly_complete] = len(counts)
            complete |= newly_complete
        # At the grid endpoint return remaining trajectories, without claiming
        # that their final selection satisfied the entropy bound.
        per_sample[~complete] = len(counts)
        return state, {
            "nfe": len(counts),
            "nfe_per_sample": per_sample.cpu().tolist(),
            "accepted_per_step": counts,
            "entropy_bound": self.bound,
            "acceptance_policy": "frozen" if self.freeze else "step_local",
            "stopping_rule": "all_committed" if self.freeze else "all_selected_same_step",
            "completed_per_sample": complete.cpu().tolist(),
        }

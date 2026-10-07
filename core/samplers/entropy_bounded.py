import math
import torch

from .utils import initialize_state
from core.modifiers.self_conditioning.session import prediction_session


def select_prefix(entropy, active, bound):
    selected = torch.zeros_like(active)
    for row in range(len(active)):
        positions = active[row].nonzero(as_tuple=True)[0]
        ordered = positions[entropy[row, positions].argsort()]
        values = entropy[row, ordered]
        if len(values):
            tolerance = torch.finfo(values.dtype).eps * max(1.0, abs(bound)) * 8
            count = (
                (values.cumsum(0) - values.cummax(0).values) <= bound + tolerance
            ).sum()
            selected[row, ordered[: int(count)]] = True
    return selected


class EntropyBoundedSampler:
    def __init__(self, entropy_bound=0.1, max_steps=180, final_denoise="error"):
        if not math.isfinite(entropy_bound) or entropy_bound < 0 or max_steps <= 0:
            raise ValueError(
                "finite nonnegative entropy bound and positive max_steps required"
            )
        self.bound, self.max_steps = entropy_bound, max_steps
        if final_denoise not in {"error", "greedy"}:
            raise ValueError("final_denoise must be error or greedy")
        self.final_denoise = final_denoise

    def sample(self, algo, prompt_tokens, prompt_mask):
        algo = prediction_session(algo)
        state = initialize_state(algo, prompt_tokens, prompt_mask)
        generated_count = state.eq(algo.process.mask_index).sum(-1).clamp_min(1)
        nfe = 0
        per_sample = torch.zeros(len(state), dtype=torch.long, device=state.device)
        accepted = []
        while True:
            active = state.eq(algo.process.mask_index)
            if not active.any():
                break
            times = active.sum(-1) / generated_count
            keep, _ = algo.schedule(times)
            prediction = algo.sampling_prediction(state, keep)
            entropy = algo.predictor.entropy(prediction)
            selected = select_prefix(entropy, active, self.bound)
            probabilities = algo.predictor.probabilities(
                prediction.float(), apply_commitment=True
            )
            state = algo.process.commit(
                state,
                probabilities,
                selected,
                deterministic=algo.predictor.commitment == "greedy",
            )
            nfe += 1
            accepted.append(selected.sum(-1).cpu().tolist())
            complete = ~(active & ~selected).any(-1)
            per_sample[(per_sample == 0) & complete & active.any(-1)] = nfe
            if complete.all():
                break
            if nfe >= self.max_steps:
                if self.final_denoise == "error":
                    raise RuntimeError(
                        "entropy-bounded sampling exceeded max_steps before completion"
                    )
                active = state.eq(algo.process.mask_index)
                keep, _ = algo.schedule(active.sum(-1) / generated_count)
                prediction = algo.sampling_prediction(state, keep)
                probabilities = algo.predictor.probabilities(
                    prediction.float(), apply_commitment=True
                )
                state = algo.process.commit(
                    state, probabilities, active, deterministic=True
                )
                nfe += 1
                per_sample[active.any(-1)] = nfe
                accepted.append(active.sum(-1).cpu().tolist())
                break
        return state, {
            "nfe": nfe,
            "nfe_per_sample": per_sample.cpu().tolist(),
            "accepted_per_step": accepted,
            "entropy_bound": self.bound,
        }

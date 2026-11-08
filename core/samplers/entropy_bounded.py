import math
import torch

from .utils import initialize_state
from core.modifiers.self_conditioning.session import prediction_session
from . import requirements


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
    def __init__(self, entropy_bound=0.1, max_steps=180):
        if not math.isfinite(entropy_bound) or entropy_bound < 0 or max_steps <= 0:
            raise ValueError(
                "finite nonnegative entropy bound and positive max_steps required"
            )
        self.bound, self.max_steps = entropy_bound, max_steps

    def sample(self, algo, prompt_tokens, prompt_mask):
        views = getattr(algo, "views", None)
        if views is None:
            requirements.process_method(algo, "reverse", "entropy-bounded sampling", "absorbing")
        algo = prediction_session(algo)
        if views is None:
            process = algo.process
            state = initialize_state(algo, prompt_tokens, prompt_mask)

            def predict(state, times):
                keep, _ = algo.schedule(times)
                return algo.sampling_prediction(state, keep)

        else:
            # The same loop on a one-view (absorbing) recipe: time is the masked
            # fraction, passed through the view's points.
            from core.processes.absorbing import AbsorbingProcess

            if len(views) != 1 or not isinstance(views[0].process, AbsorbingProcess):
                raise TypeError("entropy-bounded sampling needs one absorbing view")
            process = views[0].process
            # It commits argmax tokens, so the posterior must never choose the mask.
            requirements.require(
                getattr(algo.predictor, "mask_index", None) == process.mask_index,
                "entropy-bounded sampling needs a posterior that excludes the mask token "
                "(parameterization.mask_index); this recipe's posterior can choose it",
            )
            if prompt_tokens.ndim != 2 or prompt_tokens.shape != prompt_mask.shape:
                raise ValueError("prompt tensors must have matching [batch, length] shapes")
            if torch.any(prompt_tokens[prompt_mask.bool()] == process.mask_index):
                raise ValueError("fixed prompt cannot contain the absorbing mask")
            state = process.prior(
                prompt_tokens.shape, prompt_tokens.device, prompt_tokens, prompt_mask
            )
            generated = ~prompt_mask.bool()

            def predict(state, times):
                points = algo.points(times, corruption_mask=generated)
                return algo.sampling_prediction((state,), points, times)

        generated_count = state.eq(process.mask_index).sum(-1).clamp_min(1)
        nfe = 0
        per_sample = torch.zeros(len(state), dtype=torch.long, device=state.device)
        accepted = []
        while True:
            active = state.eq(process.mask_index)
            if not active.any():
                break
            times = active.sum(-1) / generated_count
            prediction = predict(state, times)
            entropy = algo.predictor.entropy(prediction)
            selected = select_prefix(entropy, active, self.bound)
            probabilities = algo.predictor.probabilities(
                prediction.float(), apply_commitment=True
            )
            state = process.commit(
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
                raise RuntimeError(
                    "entropy-bounded sampling exceeded max_steps before completion"
                )
        return state, {
            "nfe": nfe,
            "nfe_per_sample": per_sample.cpu().tolist(),
            "accepted_per_step": accepted,
            "entropy_bound": self.bound,
        }

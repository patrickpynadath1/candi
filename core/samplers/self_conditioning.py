"""Apply fresh two-pass self-conditioning to a sampler with trajectory sessions."""
from contextlib import nullcontext
from core.modifiers.self_conditioning.session import PredictionSession
from core.modifiers.self_conditioning.prediction import TrajectoryState


class BootstrapSession(PredictionSession):
    def _predict(self, method, *args):
        # Every field evaluation bootstraps at its own state, including Heun stages.
        self.state = TrajectoryState()
        call = getattr(self.algorithm, method)
        call(*args, trajectory=self.state)
        if self.prompt_mask is not None and self.prediction.source == "posterior_mean":
            import torch
            clean = self.representation.encode(self.prompt_tokens)
            self.state.conditioning = torch.where(
                self.prompt_mask.bool()[..., None], clean, self.state.conditioning
            )
        return call(*args, trajectory=self.state)

    def sampling_prediction(self, *args):
        return self._predict("sampling_prediction", *args)

    def greedy_prediction(self, *args):
        return self._predict("greedy_prediction", *args)

    def step(self):
        return nullcontext()


class BootstrapSelfConditioningSampler:
    """Wrap a one-call-per-prediction sampler; charge both bootstrap forwards."""
    def __init__(self, sampler, steps=32):
        self.sampler, self.steps = sampler, steps

    def steps_for_nfe(self, nfe):
        from .requirements import integer_nfe
        budget = integer_nfe(nfe) // 2
        if budget < 3:
            raise ValueError("bootstrap sampling requires at least six NFE")
        return getattr(self.sampler, "steps_for_nfe", lambda n: n - 1)(budget)

    def sample(self, algo, prompt_tokens, prompt_mask):
        if not getattr(getattr(algo, "prediction", None), "enabled", False):
            raise ValueError("bootstrap requires a trained self-conditioning input")
        # Never wrap the native sampler's own two-pass mode.
        if getattr(self.sampler, "self_conditioning_mode", "previous") == "bootstrap":
            raise ValueError("nested bootstrap sampling would double-count calls")
        session = BootstrapSession(algo)
        session.prompt_tokens, session.prompt_mask = prompt_tokens, prompt_mask
        self.sampler.steps = self.steps
        from .loop import LinearTimeGrid
        from dataclasses import replace
        if isinstance(getattr(self.sampler, "time_grid", None), LinearTimeGrid):
            self.sampler.time_grid = replace(self.sampler.time_grid, steps=self.steps)
        tokens, info = self.sampler.sample(session, prompt_tokens, prompt_mask)
        info = dict(info)
        info["nfe"] *= 2
        info["nfe_per_sample"] = [2 * value for value in info["nfe_per_sample"]]
        info["self_conditioning_mode"] = "bootstrap"
        return tokens, info

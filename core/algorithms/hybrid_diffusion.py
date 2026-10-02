"""Parallel-view hybrid diffusion with one shared clean-token posterior."""

import torch

from .posterior import PosteriorDiffusion


class HybridDiffusion(PosteriorDiffusion):
    def __init__(self, model, predictor, time_sampler, objective, length=180,
                 prediction=None, synchronize_committed=False):
        super().__init__(prediction)
        if not getattr(model, "supports_fused_streams", False):
            raise TypeError("HybridDiffusion requires a fused-stream hybrid model")
        if not callable(getattr(predictor, "training_posterior", None)):
            raise TypeError(
                "HybridDiffusion requires a clean-token posterior parameterization"
            )
        for branch in model.branches:
            if getattr(branch.process, "prediction_kind", "posterior") != "posterior":
                raise TypeError(
                    "hybrid branches must consume clean-token posteriors; "
                    "for uniform noise select process=uniform_posterior "
                    "(PosteriorUniformProcess), not the plug-in uniform process"
                )
        self.model, self.predictor = model, predictor
        self.time_sampler, self.objective = time_sampler, objective
        self.length = length
        self.synchronize_committed = bool(synchronize_committed)
        self.validate_prediction()

    @property
    def branches(self):
        return self.model.branches

    def points(self, global_time):
        return tuple(branch.point(global_time) for branch in self.branches)

    def model_inputs(self, states, points, global_time):
        return states, points, global_time

    def prediction_mean(self, probabilities):
        # Exactly one fused vector is fed back, regardless of branch count.
        return self.model.clean_embedding(probabilities)

    def _observed(self, states):
        for state in states:
            if state.ndim == 2 and state.dtype in (torch.int32, torch.int64):
                return state
        return torch.zeros(states[0].shape[:2], dtype=torch.long, device=states[0].device)

    def _prediction(self, pathway, states, points, global_time, trajectory=None):
        result = self.prediction.predict(
            self.model, self.predictor,
            self.model_inputs(states, points, global_time), self._observed(states),
            pathway, mean=self.prediction_mean,
            logits_transform=self.prediction_logits, trajectory=trajectory,
        )
        observed = self._observed(states)
        mask_indices = [
            getattr(branch.process, "mask_index", None) for branch in self.branches
        ]
        mask_index = next((value for value in mask_indices if value is not None), None)
        if mask_index is not None:
            committed = observed.ne(mask_index)
            fixed = torch.full_like(result, -torch.inf)
            fixed.scatter_(-1, observed.clamp_max(result.shape[-1] - 1)[..., None], 0)
            result = torch.where(committed[..., None], fixed, result)
        return result

    def training_prediction(self, states, points, global_time):
        return self._prediction("training_posterior", states, points, global_time)

    def sampling_prediction(self, states, points, global_time, trajectory=None):
        return self._prediction(
            "sampling_posterior", states, points, global_time, trajectory
        )

    def greedy_prediction(self, states, points, global_time, trajectory=None):
        return self._prediction(
            "greedy_posterior", states, points, global_time, trajectory
        )

    def loss(self, tokens, loss_mask, corruption_mask=None, *, step=0,
             training=False, **time_context):
        del step, training
        if corruption_mask is None:
            corruption_mask = loss_mask
        time = self.time_sampler.sample(len(tokens), tokens.device, **time_context)
        points = self.points(time)
        states = tuple(
            branch.corrupt(tokens, point, corruption_mask)
            for branch, point in zip(self.branches, points)
        )
        if self.synchronize_committed:
            states = self._synchronize_committed(states)
        prediction = self.training_prediction(states, points, time)
        per_token = self.objective(prediction, tokens)
        weights = loss_mask.to(per_token)
        count = weights.sum()
        if count <= 0:
            raise ValueError("loss mask must include at least one target")
        total = (per_token * weights).sum()
        return {"loss": total / count, "loss_sum": total.detach(),
                "num_tokens": count.detach()}

    def _synchronize_committed(self, states):
        discrete = next((state for state in states if state.ndim == 2), None)
        if discrete is None:
            return states
        mask_index = next(
            (getattr(branch.process, "mask_index", None) for branch in self.branches
             if getattr(branch.process, "mask_index", None) is not None), None
        )
        if mask_index is None:
            return states
        committed = discrete.ne(mask_index)
        return tuple(
            branch.clamp_committed(state, discrete, committed)
            if callable(getattr(branch, "clamp_committed", None)) else state
            for branch, state in zip(self.branches, states)
        )

    @torch.no_grad()
    def generate(self, sampler, prompt_tokens, prompt_mask):
        was_training = self.training
        self.eval()
        try:
            return sampler.sample(self, prompt_tokens, prompt_mask)
        finally:
            self.train(was_training)

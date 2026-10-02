"""Spherical flow matching training and prediction orchestration."""

import torch
from .posterior import PosteriorDiffusion
from .adaptive import validate_adaptive_feedback, observe_training_loss
from .discrete import reduce_token_loss
from core.representations.embedding import EmbeddingRepresentation


class SphericalFlowMatching(PosteriorDiffusion):
    def __init__(
        self,
        model,
        predictor,
        process,
        schedule,
        time_sampler,
        objective,
        representation,
        length=180,
        prediction=None,
    ):
        super().__init__(prediction)
        self.model, self.predictor, self.process = model, predictor, process
        self.schedule, self.time_sampler, self.objective = (
            schedule,
            time_sampler,
            objective,
        )
        self.representation, self.length = representation, length
        if (
            not isinstance(representation, EmbeddingRepresentation)
            or not representation.spec.unit_normalized
        ):
            raise ValueError(
                "spherical flow matching requires unit-normalized embeddings"
            )
        if (
            model.state_dim != representation.spec.dimension
            or model.vocab_size != representation.spec.vocab_size
        ):
            raise ValueError("model and representation dimensions must match")
        if getattr(schedule, "output_kind", None) != "keep_rate":
            raise ValueError("spherical schedule must provide keep coordinates")
        validate_adaptive_feedback(schedule, predictor)
        self.validate_prediction()

    def model_inputs(self, state, keep):
        return state, -keep.clamp_min(1e-12).log()

    def prediction_logits(self, logits):
        return logits.float()

    def prediction_mean(self, probabilities):
        return self.representation.posterior_mean(probabilities)

    def loss(
        self,
        tokens,
        loss_mask,
        corruption_mask=None,
        *,
        step=0,
        training=False,
        **time_context,
    ):
        time = self.time_sampler.sample(len(tokens), tokens.device, **time_context)
        keep, _ = self.schedule(time)
        state = self.process.corrupt(
            self.representation.encode(tokens),
            keep,
            loss_mask if corruption_mask is None else corruption_mask,
        )
        prediction = self.training_prediction(state, keep)
        per_token = self.objective(prediction, tokens)
        observe_training_loss(
            self.schedule,
            time,
            prediction,
            tokens,
            per_token,
            loss_mask,
            step=step,
            training=training,
        )
        return reduce_token_loss(per_token, loss_mask)

    @torch.no_grad()
    def generate(self, sampler, prompt_tokens, prompt_mask):
        was_training = self.training
        self.eval()
        try:
            return sampler.sample(self, prompt_tokens, prompt_mask)
        finally:
            self.train(was_training)

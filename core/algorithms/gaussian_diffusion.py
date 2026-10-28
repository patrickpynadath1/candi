"""Training and posterior-mean prediction for Gaussian diffusion over discrete representations."""

from core.representations.one_hot import OneHotRepresentation
from core.objectives.weighting import ObjectiveContext

from .posterior import PosteriorDiffusion
from .adaptive import validate_adaptive_feedback, observe_training_loss


class GaussianDiffusion(PosteriorDiffusion):
    def __init__(
        self,
        model,
        predictor,
        process,
        schedule,
        time_sampler,
        objective,
        length=180,
        representation=None,
        prediction=None,
    ):
        super().__init__(prediction)
        self.model, self.predictor, self.process = model, predictor, process
        self.schedule, self.time_sampler, self.objective = (
            schedule,
            time_sampler,
            objective,
        )
        self.length = length
        validate_adaptive_feedback(schedule, predictor)
        # The default also reconstructs saved pre-representation one-hot configs.
        self.representation = (
            representation
            if representation is not None
            else OneHotRepresentation(model.vocab_size)
        )
        self._validate_schedule_representation()
        self.validate_prediction()

    def _validate_schedule_representation(self):
        spec = self.representation.spec
        self.schedule.representation_requirements.validate(
            spec, type(self.schedule).__name__
        )
        if self.model.state_dim != spec.dimension:
            raise ValueError("model state_dim must match representation dimension")
        if self.model.vocab_size != spec.vocab_size:
            raise ValueError("model vocabulary must match representation vocabulary")

    def model_inputs(self, state, time, path):
        return state, time, path.sigma, path.alpha

    def prediction_mean(self, probabilities):
        return self.representation.posterior_mean(probabilities)

    def posterior_mean(self, state, time, path):
        probabilities = self.sampling_prediction(state, time, path).float().exp()
        return self.representation.posterior_mean(probabilities)

    def loss(
        self,
        tokens,
        loss_mask,
        corruption_mask=None,
        *,
        step=0,
        training=False,
        time_sampler=None,
        objective=None,
        **time_context,
    ):
        time_sampler = self.time_sampler if time_sampler is None else time_sampler
        objective = self.objective if objective is None else objective
        if corruption_mask is None:
            corruption_mask = loss_mask
        time = self.schedule.training_time(
            time_sampler.sample(len(tokens), tokens.device, **time_context)
        )
        path = self.schedule(time)
        clean = self.representation.encode(tokens)
        state = self.process.corrupt(clean, path.alpha, path.sigma, corruption_mask)
        prediction = self.training_prediction(state, time, path)
        if getattr(objective, "requires_context", False):
            per_token = objective(
                prediction,
                tokens,
                context=ObjectiveContext(time=time, alpha=path.alpha, sigma=path.sigma),
            )
        else:
            per_token = objective(prediction, tokens)
        weights = loss_mask.to(per_token)
        count = weights.sum()
        if count <= 0:
            raise ValueError("loss mask must include at least one target")
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
        total = (per_token * weights).sum()
        return {
            "loss": total / count,
            "loss_sum": total.detach(),
            "num_tokens": count.detach(),
        }

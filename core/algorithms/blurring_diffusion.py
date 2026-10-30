"""Training orchestration for separable positional/feature blurring diffusion."""

import torch
from core.sequences import resolve_sequence_layout
from .adaptive import observe_training_loss
from .posterior import PosteriorDiffusion


class BlurringDiffusion(PosteriorDiffusion):
    supports_sequence_layout = True

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
        detach_forward_embedding=False,
    ):
        super().__init__(prediction)
        self.model, self.predictor, self.process = model, predictor, process
        self.schedule, self.time_sampler, self.objective = (
            schedule,
            time_sampler,
            objective,
        )
        self.representation, self.length = representation, length
        self.detach_forward_embedding = bool(detach_forward_embedding)
        if getattr(objective, "requires_context", False):
            raise TypeError(
                "blurring diffusion supports token cross entropy without scalar Gaussian weighting"
            )
        if model.state_dim != representation.spec.dimension:
            raise ValueError("model state_dim must match blur representation dimension")
        if model.vocab_size != representation.spec.vocab_size:
            raise ValueError("model and representation vocabularies must match")
        if model.input_preconditioning != "none" or model.time_conditioning != "time":
            raise ValueError(
                "blurring diffusion requires raw time and no input preconditioning"
            )
        self.validate_prediction()

    def model_inputs(self, state, time):
        # ContinuousDiT's generic interface accepts path scalars. These are not
        # used by the configured raw-time/no-preconditioning architecture.
        return state, time, torch.ones_like(time), torch.ones_like(time)

    def prediction_mean(self, probabilities):
        return self.representation.posterior_mean(probabilities)

    def loss(
        self,
        tokens,
        loss_mask,
        corruption_mask=None,
        *,
        sequence_layout=None,
        step=0,
        training=False,
        **time_context,
    ):
        if corruption_mask is None:
            corruption_mask = loss_mask
        time = self.time_sampler.sample(len(tokens), tokens.device, **time_context)
        clean = self.representation.encode(tokens)
        process_clean = clean.detach() if self.detach_forward_embedding else clean
        layout = resolve_sequence_layout(corruption_mask, sequence_layout)
        path = self.schedule.for_layout(
            time.to(clean), corruption_mask, clean.shape[-1], layout
        )
        state = self.process.corrupt(process_clean, path)
        prediction = self.training_prediction(state, time)
        per_token = self.objective(prediction, tokens)
        weights = loss_mask.to(per_token)
        count = weights.sum()
        if count <= 0:
            raise ValueError("loss mask must include at least one target")
        observe = getattr(self.time_sampler, "observe", None)
        if training and observe is not None:
            observe(time, per_token, loss_mask, step)
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

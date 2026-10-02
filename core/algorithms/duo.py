"""DUO curriculum orchestration over uniform diffusion and Gaussian relaxation."""

from .uniform_diffusion import UniformDiffusion
from .discrete import reduce_token_loss
from .adaptive import observe_training_loss
from core.representations.one_hot import OneHotRepresentation
from core.processes.gaussian import GaussianProcess


class Duo(UniformDiffusion):
    def __init__(
        self,
        *args,
        calibration,
        curriculum,
        relaxation_process,
        representation,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.calibration, self.curriculum = calibration, curriculum
        self.relaxation_process, self.representation = (
            relaxation_process,
            representation,
        )
        if not isinstance(representation, OneHotRepresentation) or not isinstance(
            relaxation_process, GaussianProcess
        ):
            raise TypeError(
                "DUO relaxation requires one-hot encoding and Gaussian corruption"
            )
        if (
            calibration.vocab_size != self.process.vocab_size
            or representation.spec.vocab_size != self.process.vocab_size
        ):
            raise ValueError(
                "DUO calibration/representation vocabulary must match uniform process"
            )

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
        if not self.curriculum.use_relaxation(step, training):
            return super().loss(
                tokens,
                loss_mask,
                corruption_mask,
                step=step,
                training=training,
                **time_context,
            )
        if corruption_mask is None:
            corruption_mask = loss_mask
        times = self.time_sampler.sample(len(tokens), tokens.device, **time_context)
        keep, derivative = self.schedule(times)
        noise = self.calibration(keep, derivative)
        clean = self.representation.encode(tokens)
        relaxed = self.relaxation_process.corrupt(
            clean, noise.alpha, noise.sigma, corruption_mask
        )
        relaxed = relaxed * self.curriculum.inverse_temperature(step)
        # The model's soft-input embedding applies softmax to these scaled
        # scores; the plug-in predictor separately normalizes output logits.
        discrete_state = relaxed.argmax(-1)
        prediction = self.training_prediction(relaxed, noise.keep)
        losses = self.objective(
            prediction,
            discrete_state,
            tokens,
            noise.keep[:, None],
            noise.derivative[:, None],
        )
        observe_training_loss(
            self.schedule,
            times,
            prediction,
            tokens,
            losses,
            loss_mask,
            step=step,
            training=training,
        )
        return reduce_token_loss(losses, loss_mask)

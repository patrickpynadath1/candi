"""DUO curriculum orchestration over uniform diffusion and Gaussian relaxation.

Outside the curriculum (after ``end_step``, or when not training) DUO is the
plain uniform recipe. Inside it, the network reads a softened Gaussian
relaxation of the one-hot tokens instead of a uniform-corrupted token state.
"""

from torch.nn import functional as F

from .uniform_diffusion import UniformDiffusion
from .adaptive import observe_training_loss
from .training_loss import reduce_token_loss
from core.modifiers.hybrid.views import ViewPoint
from core.processes.gaussian import GaussianProcess
from core.representations.noisy_inputs import TokenRepresentation
from core.representations.one_hot import OneHotRepresentation


class SoftTokenRepresentation(TokenRepresentation):
    """Token ids, or [B, L, V] scores embedded as their softmax expectation.

    The scores branch is the DiT ``EmbeddingLayer``'s soft input: softmax and
    the expected embedding in FP32, returned in the scores' dtype.
    """

    def to_model_embedding(self, state, point):
        if state.ndim == 2:
            return super().to_model_embedding(state, point)
        if state.ndim != 3:
            raise ValueError("soft token states must be [batch, length, vocab]")
        expected = F.softmax(state, dim=-1).float() @ self._table().float()
        return expected.to(state.dtype) * self.scale


class Duo(UniformDiffusion):
    representation_type = SoftTokenRepresentation

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
        # ``representation`` names the uniform view's token representation
        # (OneViewAlgorithm); the one-hot relaxation encoding is separate.
        self.relaxation_process = relaxation_process
        self.relaxation_representation = representation
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
        sequence_layout=None,
        **time_context,
    ):
        if not self.curriculum.use_relaxation(step, training):
            return super().loss(
                tokens,
                loss_mask,
                corruption_mask,
                step=step,
                training=training,
                sequence_layout=sequence_layout,
                **time_context,
            )
        if corruption_mask is None:
            corruption_mask = loss_mask
        time = self.time_sampler.sample(len(tokens), tokens.device, **time_context)
        (local,) = self.points(
            time,
            training=True,
            corruption_mask=corruption_mask,
            sequence_layout=sequence_layout,
        )
        noise = self.calibration(*local.coefficients)
        relaxed = self.relaxation_process.corrupt(
            self.relaxation_representation.encode(tokens),
            noise.alpha,
            noise.sigma,
            corruption_mask,
        )
        # The view's soft-input embedding applies softmax to these scaled
        # scores; the plug-in predictor separately normalizes output logits.
        relaxed = relaxed * self.curriculum.inverse_temperature(step)
        # Time conditioning and the objective use the calibrated keep rate.
        calibrated = (ViewPoint(local.time, (noise.keep, noise.derivative)),)
        prediction = self.training_prediction((relaxed,), calibrated, time)
        per_token = self._objective(
            self.objective, prediction, tokens, time, calibrated, (relaxed.argmax(-1),)
        )
        # Only the schedule observes relaxed losses; the time sampler does not.
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

"""Conditional hybrid ELBO orchestration over coupled Gaussian/absorbing views."""
import torch
from core.modifiers.hybrid.training import HybridDiffusion
from core.modifiers.hybrid.views import DiffusionView, ViewPoint
from core.models.revealed_context import RevealedContextInput
from core.noise_schedules.shared_clock import ClockGaussianSchedule, ClockKeepSchedule
from core.noise_schedules.gaussian_path import path_from_gamma
from core.processes.absorbing import AbsorbingProcess
from core.representations.tokens import TokenRepresentation
from core.samplers.hybrid_updates import GaussianPosteriorUpdate, DiscretePosteriorUpdate


class RePlaidHybrid(HybridDiffusion):
    def __init__(self, model, predictor, process, clock, representation, objective,
                 corruption, time_sampler, length=512, prediction=None, context_scale="unit_embedding", context_input=None):
        if model.self_conditioning is not None:
            raise ValueError("revealed context is not posterior self-conditioning")
        if not representation.spec.unit_normalized or model.state_dim != representation.spec.dimension:
            raise ValueError("hybrid ELBO requires matching unit-normalized embeddings")
        views = [
            DiffusionView(process, ClockGaussianSchedule(clock), representation, GaussianPosteriorUpdate()),
            DiffusionView(AbsorbingProcess(model.vocab_size), ClockKeepSchedule(clock),
                          TokenRepresentation(model.vocab_size), DiscretePosteriorUpdate()),
        ]
        factory = RevealedContextInput if context_input is None else context_input
        model.input = factory(views=views, continuous_input=model.input,
                              hidden_size=model.input.projection.out_features, context_scale=context_scale)
        super().__init__(model, predictor, time_sampler, objective, length, prediction,
                         synchronize_committed=True)
        self.corruption = corruption

    @property
    def clock(self):
        return self.views[0].schedule.clock

    @property
    def representation(self):
        return self.views[0].representation

    def prediction_mean(self, probabilities):
        return self.representation.posterior_mean(probabilities)

    def prediction_transform(self, states, points):
        path = points[0].coefficients
        return lambda logits: self.predictor.corrected_logits(
            logits, states[0], path.alpha, path.sigma, self.representation.codebook())

    def loss(self, tokens, loss_mask, corruption_mask=None, *, training=False, **context):
        if context.get("sequence_layout") is not None:
            raise ValueError("conditional hybrid ELBO does not support block sequence layouts")
        if not torch.all((loss_mask == 0) | (loss_mask == 1)):
            raise ValueError("hybrid ELBO requires binary loss masks")
        valid = loss_mask.bool()
        if corruption_mask is not None and not torch.equal(corruption_mask.bool(), valid):
            raise ValueError("hybrid conditional ELBO requires corruption_mask == loss_mask")
        if not valid.any():
            raise ValueError("loss mask must contain targets")
        target = self.corruption.select_target(valid)
        value, density = self.clock.draw(len(tokens))
        gamma = self.clock.gamma_at(value)
        lower, upper = self.clock.bounds()
        path = path_from_gamma(gamma, torch.ones_like(gamma)*(upper-lower))
        clean = self.representation.encode(tokens)
        masked = self.corruption(valid, value, target=target)
        state = self.views[0].process.corrupt(clean, path.alpha, path.sigma, masked)
        discrete = torch.where(masked, self.views[1].process.mask_index, tokens)
        points = (ViewPoint(value, path), ViewPoint(value, (1-value, -torch.ones_like(value))))
        log_probs = self.training_prediction((state, discrete), points, value)
        predicted = self.prediction_mean(log_probs.exp())
        per_token, estimate = self.objective(log_probs, tokens, clean, predicted, gamma,
                                           value, density, lower, upper, masked, valid)
        if training:
            per_token = per_token + self.clock.variance_surrogate(estimate, density)[:, None]*valid
        total = per_token.sum()
        return dict(loss=total/valid.sum(), loss_sum=total.detach(), num_tokens=valid.sum().detach())

"""Discrete-time absorbing diffusion (D3PM) trained with its variational bound.

One absorbing view bound to the configured token network's embedding table.
"""

import torch

from .discrete import absorbing_token_network, validate_discrete_recipe
from core.modifiers.hybrid.one_view import OneViewAlgorithm
from core.modifiers.hybrid.training import HybridDiffusion
from core.noise_schedules.log_linear import LogLinear
from core.parameterizations.posterior import PosteriorParameterization
from core.parameterizations.constraints import UnconstrainedPosterior, TokenExclusion
from core.objectives.d3pm import D3PMAbsorbingObjective
from core.processes.absorbing import AbsorbingProcess
from core.samplers.hybrid_updates import DiscretePosteriorUpdate
from core.time_sampling.discrete import DiscreteTimeSampler


class D3PMAbsorb(OneViewAlgorithm, HybridDiffusion):
    # Training never forces observed tokens to one-hot: the bound needs the
    # predicted mask probability and a nonzero reconstruction loss at clean
    # inputs. Generation pathways keep observed tokens (see _prediction).
    clamp_observed = False
    # The loss runs the network twice: the noisy state and the clean reconstruction.
    training_passes = 2

    def __init__(
        self,
        model,
        predictor,
        process,
        schedule,
        time_sampler,
        objective,
        length=180,
        prediction=None,
    ):
        validate_discrete_recipe(model, predictor, schedule, prediction)
        if not isinstance(schedule, LogLinear):
            raise TypeError("D3PM objective requires a log-linear keep schedule")
        if not isinstance(predictor, PosteriorParameterization) or not isinstance(
            predictor.constraints, (UnconstrainedPosterior, TokenExclusion)
        ):
            raise TypeError(
                "D3PM requires token probabilities without observed-token clamping"
            )
        if (
            isinstance(predictor.constraints, TokenExclusion)
            and predictor.constraints.token_id == process.mask_index
        ):
            raise ValueError(
                "D3PM cannot exclude its mask token; the bound needs its probability"
            )
        if not isinstance(process, AbsorbingProcess) or not isinstance(
            objective, D3PMAbsorbingObjective
        ):
            raise TypeError("D3PM requires absorbing process and D3PM objective")
        if (
            not isinstance(time_sampler, DiscreteTimeSampler)
            or time_sampler.steps != objective.steps
        ):
            raise ValueError("D3PM time sampler and objective step counts must match")
        if objective.mask_index != process.mask_index:
            raise ValueError("D3PM mask indices must match")
        network = absorbing_token_network(
            model, process, schedule, DiscretePosteriorUpdate()
        )
        super().__init__(
            network, predictor, time_sampler, objective, length=length, prediction=prediction
        )

    def _prediction(self, pathway, states, points, global_time, trajectory=None):
        result = super()._prediction(
            pathway, states, points, global_time, trajectory=trajectory
        )
        if pathway == self._pathway("training"):
            return result
        # Generation keeps observed tokens, as the standalone samplers did by
        # updating only masked positions (e.g. greedy terminal decoding).
        observed, committed = self.process.observed_tokens(states[0])
        fixed = torch.full_like(result, -torch.inf).scatter_(
            -1, observed.clamp_max(result.shape[-1] - 1)[..., None], 0
        )
        return torch.where(committed[..., None], fixed, result)

    def _objective(self, objective, prediction, tokens, time, points, states, **entries):
        # The bound's reconstruction term: a second pass on the clean tokens
        # at the schedule's clean endpoint.
        clean_time = torch.zeros_like(time)
        reconstruction = self.training_prediction(
            (tokens,), self.points(clean_time, training=True), clean_time
        )
        keep = points[0].coefficients[0]
        return objective(prediction, states[0], tokens, keep, reconstruction)

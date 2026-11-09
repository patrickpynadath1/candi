"""SEDD absorbing: score-entropy training of absorbing log scores.

One absorbing view bound to the configured token network's embedding table, with
the parameterization's score pathway.
"""

import torch

from .discrete import absorbing_token_network, keep_rate_objective, validate_discrete_recipe
from core.modifiers.hybrid.one_view import OneViewAlgorithm
from core.modifiers.hybrid.training import HybridDiffusion
from core.parameterizations.score import AbsorbingScoreParameterization
from core.processes.absorbing import AbsorbingProcess
from core.objectives.score_entropy import AbsorbingScoreEntropy
from core.samplers.analytic import AbsorbingScoreUpdate


class SeddAbsorb(OneViewAlgorithm, HybridDiffusion):
    prediction_pathway = "score"
    # The process kind check stays "posterior": AbsorbingProcess declares no
    # prediction kind, and its view serves either a posterior or a score
    # (the update rule decides which it consumes).
    prediction_kind = "posterior"
    # The parameterization scatters a zero log score at the observed token.
    clamp_observed = False

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
        if not isinstance(predictor, AbsorbingScoreParameterization):
            raise TypeError("SeddAbsorb requires AbsorbingScoreParameterization")
        if not isinstance(process, AbsorbingProcess) or not isinstance(
            objective, AbsorbingScoreEntropy
        ):
            raise TypeError(
                "SeddAbsorb requires an absorbing process and score-entropy objective"
            )
        if (
            objective.mask_index != process.mask_index
            or objective.vocab_size != model.output_layer.linear.out_features
        ):
            raise ValueError(
                "SEDD model, process and objective token spaces must agree"
            )
        network = absorbing_token_network(model, process, schedule, AbsorbingScoreUpdate())
        super().__init__(
            network, predictor, time_sampler, objective, length=length, prediction=prediction
        )

    def generate(self, sampler, prompt_tokens, prompt_mask):
        rule = getattr(sampler, "update_rule", None)
        if getattr(rule, "prediction_pathway", "posterior") != self.prediction_pathway:
            raise TypeError(
                f"{type(sampler).__name__} requires PosteriorParameterization; "
                "SEDD log scores need a score sampler (inference/sampler=analytic)"
            )
        return super().generate(sampler, prompt_tokens, prompt_mask)

    def _prediction(self, pathway, states, points, global_time, trajectory=None):
        # Scores are scaled by the noise level -log(keep) the network sees.
        keep = points[0].coefficients[0]
        sigma = -keep.clamp_min(torch.finfo(keep.dtype).tiny).log()
        return self.prediction.predict(
            self.model,
            self.predictor.at_noise(sigma),
            self.model_inputs(states, points, global_time),
            self._observed(states),
            pathway,
            logits_transform=self.prediction_logits,
            trajectory=trajectory,
            training_pathway=self._pathway("training"),
        )

    def _objective(self, objective, prediction, tokens, time, points, states, **entries):
        keep, derivative = points[0].coefficients  # CE may come as a per-call override
        return keep_rate_objective(objective, prediction, states[0], tokens, time, keep, derivative)

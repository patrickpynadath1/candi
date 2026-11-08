"""Simplex diffusion: Dirichlet corruption trained as clean-token posterior prediction.

One Dirichlet view held by the algorithm; the network reads the raw sampler time.
The process and sampler own the transition mathematics.
"""

from core.modifiers.hybrid.one_view import OneViewAlgorithm
from core.modifiers.hybrid.training import HybridDiffusion
from core.modifiers.hybrid.views import DiffusionView
from core.modifiers.self_conditioning.sources import posterior_mean
from core.representations.simplex_state import SimplexStateRepresentation
from core.samplers.hybrid_updates import SimplexPosteriorUpdate


class SimplexDiffusion(OneViewAlgorithm, HybridDiffusion):
    # Simplex states carry no observed (committed) tokens to clamp.
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
        condition_dropout=0.0,
    ):
        if not 0 <= condition_dropout <= 1:
            raise ValueError("condition_dropout must be in [0,1]")
        if model.vocab_size != process.vocab_size:
            raise ValueError("simplex process and model vocabulary must match")
        representation = SimplexStateRepresentation(process.vocab_size, process.state_dtype)
        view = DiffusionView(process, schedule, representation, SimplexPosteriorUpdate())
        model.time.coordinate = "time"  # The raw sampler time conditions the network.
        network = model
        super().__init__(
            network,
            predictor,
            time_sampler,
            objective,
            length=length,
            prediction=prediction,
            views=[view],
        )
        self.condition_dropout = condition_dropout
        self._drop_condition = False

    def prediction_logits(self, logits):
        return logits.float()

    def prediction_embedding_table(self):
        return self.model.input.table

    def prediction_mean(self, probabilities):
        # FP32 matmul outside autocast against the network's token table.
        return posterior_mean(probabilities, self.prediction_embedding_table())

    def _observed(self, states):
        # The parameterization and self-conditioning see the simplex state.
        return states[0]

    def _corrupt(self, tokens, points, corruption_mask):
        state = self.process.corrupt(tokens, points[0].coefficients, corruption_mask)
        if self._drop_condition:
            state = self.process.drop_condition(state, corruption_mask, self.condition_dropout)
        return (state,)

    def _objective(self, objective, prediction, tokens, time, points, states):
        return objective(prediction, tokens)

    def _observe(self, sampler, feedback_time, time, points, prediction, tokens, per_token, loss_mask, step):
        # Only the time sampler adapts; the simplex schedule takes no feedback.
        observe = getattr(sampler, "observe", None)
        if observe is not None:
            observe(feedback_time, per_token, loss_mask, step)

    def loss(self, tokens, loss_mask, corruption_mask=None, *, training=False, **options):
        """Condition dropout corrupts prompts only for training losses."""
        self._drop_condition = training
        try:
            return super().loss(tokens, loss_mask, corruption_mask, training=training, **options)
        finally:
            self._drop_condition = False

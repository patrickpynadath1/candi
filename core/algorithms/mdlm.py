"""MDLM: absorbing (masked) diffusion trained as clean-token posterior prediction.

One absorbing view bound to the configured token network's embedding table. With
``time_conditioning`` the network reads -log(keep) (keep clamped at the dtype's
smallest normal); otherwise a zero time input, keeping any AdaLN modulation.
"""


from core.modifiers.hybrid.one_view import OneViewAlgorithm
from core.modifiers.hybrid.training import HybridDiffusion
from core.modifiers.hybrid.views import DiffusionView
from core.representations.noisy_inputs import TokenRepresentation
from core.samplers.hybrid_updates import DiscretePosteriorUpdate


class MDLM(OneViewAlgorithm, HybridDiffusion):
    def __init__(
        self,
        model,
        predictor,
        process,
        schedule,
        time_sampler,
        objective,
        length=180,
        time_conditioning=False,
        prediction=None,
    ):
        table = model.token_table
        representation = TokenRepresentation(*table.shape, embedding=table)
        view = DiffusionView(process, schedule, representation, DiscretePosteriorUpdate())
        model.time.coordinate = "negative_log_keep_exact" if time_conditioning else "zero"
        network = model.bind_views([view], fp32_clean_embedding=True)
        super().__init__(
            network, predictor, time_sampler, objective, length=length, prediction=prediction
        )
        self.time_conditioning = time_conditioning

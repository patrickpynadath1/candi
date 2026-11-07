"""MDLM: absorbing (masked) diffusion trained as clean-token posterior prediction.

Runs on the view path: one absorbing view on a ModularDiT adopted from the
configured DiT (see core/models/legacy.py:adopt). Training, sampling and
parameter order are bit-identical to the former standalone implementation,
which core/tests/reference_algorithms/mdlm.py keeps for comparison.
"""

import torch

from core.models.legacy import adopt
from core.modifiers.hybrid.training import HybridDiffusion
from core.modifiers.hybrid.views import DiffusionView
from core.representations.noisy_inputs import TokenRepresentation
from core.samplers.hybrid_updates import DiscretePosteriorUpdate


class MDLM(HybridDiffusion):
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
        table = model.vocab_embed.embedding
        with torch.random.fork_rng(devices=[]):  # The table is adopted below.
            representation = TokenRepresentation(*table.shape)
        view = DiffusionView(process, schedule, representation, DiscretePosteriorUpdate())
        network = adopt(
            model,
            views=[view],
            # -log(keep) exactly as MDLM computed it, or zero time input.
            time_coordinate="negative_log_keep_exact" if time_conditioning else "zero",
            fp32_clean_embedding=True,
        )
        super().__init__(
            network, predictor, time_sampler, objective, length=length, prediction=prediction
        )
        self.time_conditioning = time_conditioning

    @property
    def process(self):
        return self.views[0].process

    @property
    def schedule(self):
        return self.views[0].schedule

    def __setattr__(self, name, value):
        # nn.Module routes module values before property setters; send the
        # view's components to the view.
        if name in ("process", "schedule") and "model" in self._modules:
            setattr(self.views[0], name, value)
        else:
            super().__setattr__(name, value)

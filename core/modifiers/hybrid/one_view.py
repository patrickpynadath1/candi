"""Run a standalone single-process recipe on the hybrid path as one view.

A standalone recipe (backbone, process, schedule, ...) and its one-view hybrid
compute the same thing. ``one_view_model`` wraps the standalone backbone without
creating or reinitializing parameters, so training from scratch, saved
checkpoints (via ``HybridModel``'s load hook) and EMA parameter order are all
unchanged.
"""

from core.representations.noisy_inputs import TokenRepresentation
from core.samplers.hybrid_updates import DiscretePosteriorUpdate

from .model import HybridModel
from .views import DiffusionView


def one_view_model(
    backbone,
    process,
    schedule,
    *,
    representation=None,
    update=None,
    time_conditioning="negative_log_keep",
    name=None,
):
    """Wrap ``backbone`` as a one-view ``HybridModel`` reusing its parameters.

    The token view adopts the backbone's own embedding table, and the
    backbone's self-conditioning adapter moves to the hybrid level, where the
    hybrid applies it at the same point (after embedding, outside autocast).
    """
    if representation is None:
        table = backbone.vocab_embed.embedding
        representation = TokenRepresentation(*table.shape, embedding=table)
    view = DiffusionView(
        process, schedule, representation, update or DiscretePosteriorUpdate(), name=name
    )
    adapter = getattr(backbone, "conditioning_adapter", None)
    backbone.conditioning_adapter = None
    model = HybridModel(backbone, [view], time_conditioning=time_conditioning)
    # Registered after the backbone and view, as it was last in the backbone,
    # so parameter (and therefore EMA and optimizer state) order is unchanged.
    model.conditioning_adapter = adapter
    return model

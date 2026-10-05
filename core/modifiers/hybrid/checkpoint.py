"""Upgrade saved hybrid configuration data; no deprecated import aliases."""


def upgrade_config(value):
    """Translate the two pre-modifier hybrid assemblies at the load boundary."""
    if isinstance(value, list):
        return [upgrade_config(item) for item in value]
    if not isinstance(value, dict):
        return value
    value = dict(value)
    target = value.get("_target_", "")
    renames = {
        "core.algorithms.hybrid_diffusion.HybridDiffusion": "core.modifiers.hybrid.training.HybridDiffusion",
        "core.models.hybrid.HybridModel": "core.modifiers.hybrid.model.HybridModel",
        "core.models.hybrid.CANDIFusion": "core.modifiers.hybrid.model.CANDIFusion",
        "core.models.hybrid.TokenEmbeddingAdapter": "core.representations.noisy_inputs.TokenRepresentation",
        "core.models.hybrid.SimplexInputAdapter": "core.representations.noisy_inputs.SimplexRepresentation",
        "core.models.hybrid.ContinuousInputAdapter": "core.representations.noisy_inputs.ProjectedRepresentation",
        "core.models.hybrid.CANDIContinuousAdapter": "core.representations.noisy_inputs.CANDIRepresentation",
        "core.samplers.hybrid_entropy_bounded.HybridEntropyBoundedSampler": "core.samplers.hybrid_entropy_bounded.CANDIEntropyBoundedSampler",
    }
    if target in renames:
        value["_target_"] = renames[target]
    if target.startswith("core.hybrid.time_maps."):
        value["_target_"] = target.replace(
            "core.hybrid.time_maps.", "core.modifiers.hybrid.time_maps."
        )
    if target == "core.models.hybrid.HybridModel":
        value["views"] = value.pop("branches")
    if target == "core.models.hybrid.CANDIFusion" and "discrete_branch" in value:
        value["discrete_view"] = value.pop("discrete_branch")
    updates = {
        "DiscreteBranch": "DiscretePosteriorUpdate",
        "GaussianBranch": "GaussianPosteriorUpdate",
        "SimplexBranch": "SimplexPosteriorUpdate",
    }
    if target.startswith("core.hybrid.branches."):
        update = updates[target.rsplit(".", 1)[-1]]
        representation = dict(value.pop("adapter"))
        state_representation = value.pop("representation", None)
        if representation["_target_"] == "core.models.hybrid.ContinuousInputAdapter":
            representation["state_representation"] = state_representation
        value.update(
            _target_="core.modifiers.hybrid.views.DiffusionView",
            representation=representation,
            update={"_target_": f"core.samplers.hybrid_updates.{update}"},
        )
    return {key: upgrade_config(item) for key, item in value.items()}

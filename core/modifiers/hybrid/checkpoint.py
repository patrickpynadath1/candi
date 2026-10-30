"""Upgrade saved hybrid configuration data; no deprecated import aliases."""


def upgrade_config(value):
    """Translate the two pre-modifier hybrid assemblies at the load boundary."""
    if isinstance(value, list):
        return [upgrade_config(item) for item in value]
    if not isinstance(value, dict):
        return value
    value = dict(value)
    target = value.get("_target_", "")
    fusion_targets = {
        "core.models.hybrid.CANDIFusion",
        "core.modifiers.hybrid.model.CANDIFusion",
    }
    # The old fusion bypassed TokenRepresentation.scale at revealed sites.
    # Resolve this from the enclosing model rather than assuming the preset's .5.
    if target in {
        "core.models.hybrid.HybridModel",
        "core.modifiers.hybrid.model.HybridModel",
    }:
        fusion = value.get("fusion")
        if isinstance(fusion, dict) and fusion.get("_target_", "") in fusion_targets:
            views = value.get("views", value.get("branches"))
            views = list(views.values()) if isinstance(views, dict) else views
            index = fusion.get("discrete_view", fusion.get("discrete_branch", 0))
            view = dict(views[index])
            representation_key = "adapter" if "adapter" in view else "representation"
            representation = dict(view[representation_key])
            representation["normalize"] = True
            view[representation_key] = representation
            original_views = value.get("views", value.get("branches"))
            if isinstance(original_views, dict):
                original_views = dict(original_views)
                original_views[list(original_views)[index]] = view
            else:
                original_views = list(original_views)
                original_views[index] = view
            value["views" if "views" in value else "branches"] = original_views
            scale = representation.get("scale", 1.0)
            if scale == 0:
                raise ValueError(
                    "cannot migrate CANDI fusion with zero discrete representation scale"
                )
            fusion = dict(fusion)
            inverse_scale = (
                {"_target_": "operator.truediv", "_args_": [1, scale]}
                if isinstance(scale, str)
                else 1 / float(scale)
            )
            fusion["weights"] = [
                {"masked": 1, "revealed": inverse_scale},
                {"masked": 1, "revealed": 0},
            ]
            value["fusion"] = fusion
    renames = {
        "core.algorithms.hybrid_diffusion.HybridDiffusion": "core.modifiers.hybrid.training.HybridDiffusion",
        "core.models.hybrid.HybridModel": "core.modifiers.hybrid.model.HybridModel",
        "core.models.hybrid.CANDIFusion": "core.modifiers.hybrid.fusion.GatedFusion",
        "core.modifiers.hybrid.model.CANDIFusion": "core.modifiers.hybrid.fusion.GatedFusion",
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
    if target in fusion_targets:
        value.setdefault(
            "weights", [{"masked": 1, "revealed": 2}, {"masked": 1, "revealed": 0}]
        )
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

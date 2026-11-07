"""Load core checkpoints or explicitly import legacy MDLM model and EMA weights."""

import torch
from . import legacy_pickle


def initialize_weights(module, path):
    """Warm-start weights, not optimizer state; resume uses Lightning."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if checkpoint.get("core_format_version") != 1:
        raise ValueError("weight initialization requires a core checkpoint")
    state = {
        key.removeprefix("algorithm."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("algorithm.")
    }
    module.algorithm.load_state_dict(state, strict=True)
    if module.ema:
        if "ema" not in checkpoint:
            raise ValueError("EMA initialization requires saved EMA")
        module.ema.load_state_dict(checkpoint["ema"])
        module.ema.copy_to(module.algorithm.parameters())


# Sections whose values change the evaluated mathematics or architecture.
RECIPE_SECTIONS = (
    "algorithm",
    "block_diffusion",
    "variable_length",
    "token_feedback",
    "encoder_decoder",
)
# Inference-only predictor knobs and training-only components may differ.
INFERENCE_KEYS = {
    "algorithm.predictor.temperature",
    "algorithm.predictor.top_k",
    "algorithm.predictor.top_p",
    "algorithm.predictor.commitment",
    "algorithm.time_sampler",
    "algorithm.objective",
}


def _differences(saved, current, prefix):
    if prefix in INFERENCE_KEYS:
        return []
    if isinstance(saved, dict) and isinstance(current, dict):
        return [
            difference
            for key in sorted(set(saved) | set(current))
            for difference in _differences(
                saved.get(key), current.get(key), f"{prefix}.{key}"
            )
        ]
    return [] if saved == current else [prefix]


def recipe_mismatches(saved, current):
    """Dotted recipe keys whose resolved values differ between two configs."""
    return [
        difference
        for section in RECIPE_SECTIONS
        for difference in _differences(
            saved.get(section), current.get(section), section
        )
    ]


def check_recipe(module, checkpoint, *, allow_mismatch=False):
    """Parameter-free math (schedule, process, time conditioning) must match too."""
    saved = (checkpoint.get("hyper_parameters") or {}).get("config")
    if saved is None:
        return []
    from omegaconf import OmegaConf
    from core.modifiers.hybrid.checkpoint import upgrade_config

    # The module's own config passed through the same upgrade at construction.
    current = OmegaConf.to_container(module.config, resolve=True)
    mismatches = recipe_mismatches(upgrade_config(saved), current)
    if mismatches and not allow_mismatch:
        raise ValueError(
            "checkpoint was trained with a different recipe: "
            + ", ".join(mismatches[:10])
            + "; set evaluation.allow_config_mismatch=true to override"
        )
    return mismatches


def load_weights(module, path, *, legacy=False, allow_config_mismatch=False):
    # Checkpoints are user-supplied trusted training artifacts, not remote input.
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
        pickle_module=legacy_pickle if legacy else None,
    )
    if legacy == "candi":
        from .candi_checkpoint import load_candi_training_module

        return load_candi_training_module(module, checkpoint)
    state = checkpoint["state_dict"]
    if legacy:
        selected = {
            key.removeprefix("backbone.").replace("_orig_mod.", ""): value
            for key, value in state.items()
            if key.startswith("backbone.")
        }
        # This nonpersistent-in-practice rotary buffer was added after the
        # published no-AdaLN baseline. It is not used by noncausal MDLM.
        selected.pop("position_rotary_emb.inv_freq", None)
        module.algorithm.model.load_state_dict(selected, strict=True)
    else:
        if checkpoint.get("core_format_version") != 1:
            raise ValueError(
                "not a core checkpoint; use evaluation.legacy=true for an MDLM import"
            )
        module.load_state_dict(state, strict=True)
        check_recipe(module, checkpoint, allow_mismatch=allow_config_mismatch)
    if module.ema:
        if "ema" not in checkpoint:
            raise ValueError(
                "checkpoint has no EMA; explicitly disable EMA to use raw weights"
            )
        module.ema.load_state_dict(checkpoint["ema"])
        parameters = [p for p in module.algorithm.parameters() if p.requires_grad]
        if any(
            p.shape != s.shape for p, s in zip(parameters, module.ema.shadow_params)
        ):
            raise ValueError(
                "EMA parameter shapes do not match the configured architecture"
            )
    return checkpoint.get("global_step", 0)

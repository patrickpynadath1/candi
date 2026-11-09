"""Load core checkpoints, or import a published external one.

``legacy`` selects the source format: ``False`` (a core checkpoint), ``True`` (a
published MDLM checkpoint, ``backbone.*`` DiT keys) or ``"candi"`` (an original
CANDI checkpoint). Imports rename keys explicitly (core/training/checkpoint_imports.py).
"""

import torch
from . import legacy_pickle
from .legacy_checkpoints import migrate_legacy_checkpoint


LEGACY_FORMATS = (False, True, "candi")


def algorithm_state(checkpoint):
    """The algorithm's weights from a Lightning checkpoint, without the ``algorithm.`` prefix."""
    return {
        key.removeprefix("algorithm."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("algorithm.")
    }


def initialize_weights(module, path):
    """Warm-start weights, not optimizer state; resume uses Lightning."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if checkpoint.get("core_format_version") != 1:
        raise ValueError("weight initialization requires a core checkpoint")
    checkpoint = migrate_legacy_checkpoint(checkpoint, module)
    module.algorithm.load_state_dict(algorithm_state(checkpoint), strict=True)
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

    current = OmegaConf.to_container(module.config, resolve=True)
    mismatches = recipe_mismatches(saved, current)
    if mismatches and not allow_mismatch:
        raise ValueError(
            "checkpoint was trained with a different recipe: "
            + ", ".join(mismatches[:10])
            + "; set evaluation.allow_config_mismatch=true to override"
        )
    return mismatches


def load_weights(module, path, *, legacy=False, allow_config_mismatch=False):
    if legacy not in LEGACY_FORMATS:
        raise ValueError(f"evaluation.legacy must be one of {LEGACY_FORMATS}")
    # Checkpoints are user-supplied trusted training artifacts, not remote input.
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
        pickle_module=legacy_pickle if legacy in (True, "candi") else None,
    )
    if legacy == "candi":
        from .candi_checkpoint import load_candi_training_module

        return load_candi_training_module(module, checkpoint)
    if legacy:
        from .checkpoint_imports import published_mdlm_state

        module.load_state_dict(published_mdlm_state(checkpoint["state_dict"]), strict=True)
    else:
        if checkpoint.get("core_format_version") != 1:
            raise ValueError(
                "not a core checkpoint; use evaluation.legacy=true for an MDLM import"
            )
        checkpoint = migrate_legacy_checkpoint(checkpoint, module)
        module.load_state_dict(checkpoint["state_dict"], strict=True)
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

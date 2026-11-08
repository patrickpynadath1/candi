"""Load core checkpoints, or explicitly convert older layouts first.

``legacy`` selects the source format: ``False`` (a current core checkpoint),
``"convert"`` (a core checkpoint saved before the view-path unification),
``True`` (a published MDLM checkpoint, ``backbone.*`` DiT keys) or
``"candi"`` (an original CANDI checkpoint). Every import goes through
core/models/legacy_checkpoints.py; nothing renames keys at load time.
"""

import torch
from . import legacy_pickle


LEGACY_FORMATS = (False, True, "convert", "candi")


def initialize_weights(module, path):
    """Warm-start weights, not optimizer state; resume uses Lightning.

    ``training.initialize_convert=true`` converts a pre-unification core
    checkpoint first (or convert it once with scripts/convert_checkpoint.py).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if checkpoint.get("core_format_version") != 1:
        raise ValueError("weight initialization requires a core checkpoint")
    if module.config.training.get("initialize_convert", False):
        from core.models.legacy_checkpoints import convert_checkpoint

        checkpoint = convert_checkpoint(checkpoint, module.config, algorithm=module.algorithm)
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


def _published_mdlm(checkpoint):
    """A published MDLM checkpoint as an old core one: its DiT is ``algorithm.model``."""
    selected = {
        "algorithm.model." + key.removeprefix("backbone.").replace("_orig_mod.", ""): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("backbone.")
    }
    # This nonpersistent-in-practice rotary buffer was added after the
    # published no-AdaLN baseline. It is not used by noncausal MDLM.
    selected.pop("algorithm.model.position_rotary_emb.inv_freq", None)
    return {**checkpoint, "state_dict": selected}


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
        from core.models.legacy_checkpoints import convert_checkpoint

        published = legacy is True
        if published:
            checkpoint = _published_mdlm(checkpoint)
        elif checkpoint.get("core_format_version") != 1:
            raise ValueError("evaluation.legacy=convert requires a core checkpoint")
        checkpoint = convert_checkpoint(checkpoint, module.config, algorithm=module.algorithm)
        module.load_state_dict(checkpoint["state_dict"], strict=True)
        if not published:
            check_recipe(module, checkpoint, allow_mismatch=allow_config_mismatch)
    else:
        if checkpoint.get("core_format_version") != 1:
            raise ValueError(
                "not a core checkpoint; use evaluation.legacy=true for an MDLM import"
            )
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

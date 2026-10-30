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


def load_weights(module, path, *, legacy=False):
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

"""Strictly import the original CANDI backbone and its positional EMA weights."""

from collections import OrderedDict

import torch


_TABLE = "backbone.vocab_embed.embedding"
_TABLE_ALIAS = "backbone.precond.embed_table.embedding"
_ROTARY = "backbone.rotary_emb.inv_freq"


def legacy_candi_state(checkpoint, *, use_ema=False):
    """Resolve the original shared parameter and exclude buffers from EMA order."""
    state = OrderedDict(
        (key.replace("_orig_mod.", ""), value)
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("backbone.")
    )
    if _TABLE not in state:
        raise ValueError("checkpoint does not contain a CANDI input embedding")
    if _TABLE_ALIAS in state and not torch.equal(state[_TABLE], state[_TABLE_ALIAS]):
        raise ValueError("CANDI checkpoint contains inconsistent shared embeddings")
    if not use_ema:
        return state
    shadows = checkpoint.get("ema", {}).get("shadow_params")
    if shadows is None:
        raise ValueError("checkpoint has no EMA; explicitly select final weights")
    # nn.Module.parameters() omits the shared table's second registration and
    # the rotary buffer. state_dict() includes both, so counting its floating
    # tensors shifts every shadow tensor after the time embedder.
    parameter_keys = [key for key in state if key not in {_TABLE_ALIAS, _ROTARY}]
    if len(parameter_keys) != len(shadows):
        raise ValueError("legacy CANDI EMA parameter count does not match backbone")
    for key, shadow in zip(parameter_keys, shadows):
        if state[key].shape != shadow.shape:
            raise ValueError(f"legacy CANDI EMA shape mismatch at {key}")
        state[key] = shadow
    if _TABLE_ALIAS in state:
        state[_TABLE_ALIAS] = state[_TABLE]
    return state


def map_legacy_candi_state(state_dict, algorithm):
    """Today's algorithm state from a ``legacy_candi_state`` result.

    The original backbone is a DiT whose keys are those of the pre-ModularDiT
    ``HybridModel`` backbone; every view with a token table shares its input
    embedding. ``convert_hybrid_state_dict`` renames that layout for the view
    ModularDiT. Every trainable tensor must come from the checkpoint; buffers
    the original did not hold (fixed by the config) keep their built values.
    """
    from core.models.legacy import convert_hybrid_state_dict

    target = OrderedDict(algorithm.state_dict())
    if not any(key.startswith("model.input.views.") for key in target):
        raise TypeError("CANDI checkpoint import requires a view-input ModularDiT")
    hybrid = {
        key: value for key, value in state_dict.items() if key not in {_TABLE_ALIAS}
    }
    for key in target:
        rest = key.removeprefix("model.input.")
        if key.startswith("model.input.views.") and rest.endswith(".representation.embedding"):
            hybrid[rest] = state_dict[_TABLE]
    converted = {
        "model." + key: value for key, value in convert_hybrid_state_dict(hybrid).items()
    }
    trainable = {
        key for key, value in algorithm.state_dict(keep_vars=True).items()
        if isinstance(value, torch.nn.Parameter) and value.requires_grad
    }
    missing = sorted(key for key in trainable if key not in converted)
    if missing:
        raise ValueError(f"CANDI import has unmapped trainable parameters: {missing}")
    for key in target:
        if key not in converted:
            continue
        value = converted[key]
        if value.shape != target[key].shape:
            raise ValueError(
                f"CANDI import shape mismatch for {key}: "
                f"checkpoint {tuple(value.shape)}, model {tuple(target[key].shape)}"
            )
        target[key] = value.to(target[key])
    return target


def load_candi_training_module(module, checkpoint):
    """Import final parameters and EMA separately for normal evaluation selection."""
    algorithm = module.algorithm
    algorithm.load_state_dict(
        map_legacy_candi_state(legacy_candi_state(checkpoint), algorithm), strict=True
    )
    if module.ema is not None:
        mapped = map_legacy_candi_state(legacy_candi_state(checkpoint, use_ema=True), algorithm)
        module.ema.load_state_dict({
            "decay": checkpoint["ema"]["decay"],
            "num_updates": checkpoint["ema"]["num_updates"],
            "shadow_params": [
                mapped[name] for name, parameter in algorithm.named_parameters()
                if parameter.requires_grad
            ],
        })
    return int(checkpoint.get("global_step", 0))

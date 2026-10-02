"""Strictly import the original CANDI backbone and its positional EMA weights."""

from collections import OrderedDict

import torch

from . import legacy_pickle


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
    """Map every trainable tensor explicitly; never retain random parameters."""
    mapper = getattr(algorithm.model, "legacy_candi_state_map", None)
    if mapper is None:
        raise TypeError("CANDI checkpoint import requires the CANDI hybrid model")
    mapping = {"model." + key: source for key, source in mapper().items()}
    parameters = dict(algorithm.named_parameters(remove_duplicate=False))
    missing = set(parameters) - set(mapping)
    if missing:
        raise ValueError(f"CANDI import has unmapped trainable parameters: {sorted(missing)}")
    target = OrderedDict(algorithm.state_dict())
    for key, source in mapping.items():
        if key not in target or source not in state_dict:
            raise ValueError(f"CANDI import is missing {source!r} for {key!r}")
        value = state_dict[source]
        if value.shape != target[key].shape:
            raise ValueError(
                f"CANDI import shape mismatch for {key}: "
                f"checkpoint {tuple(value.shape)}, model {tuple(target[key].shape)}"
            )
        target[key] = value.to(target[key])
    return target


def load_legacy_candi_checkpoint(algorithm, path, *, use_ema=True, map_location="cpu"):
    # These are trusted local training artifacts. The restricted legacy
    # unpickler avoids importing the old repository's tokenizer or packages.
    checkpoint = torch.load(
        path, map_location=map_location, weights_only=False, mmap=True,
        pickle_module=legacy_pickle,
    )
    state = legacy_candi_state(checkpoint, use_ema=use_ema)
    algorithm.load_state_dict(map_legacy_candi_state(state, algorithm), strict=True)
    return int(checkpoint.get("global_step", 0))


def load_candi_training_module(module, checkpoint):
    """Import final parameters and EMA separately for normal evaluation selection."""
    algorithm = module.algorithm
    algorithm.load_state_dict(
        map_legacy_candi_state(legacy_candi_state(checkpoint), algorithm), strict=True
    )
    if module.ema is not None:
        mapped = map_legacy_candi_state(
            legacy_candi_state(checkpoint, use_ema=True), algorithm
        )
        module.ema.load_state_dict({
            "decay": checkpoint["ema"]["decay"],
            "num_updates": checkpoint["ema"]["num_updates"],
            "shadow_params": [
                mapped[name] for name, parameter in algorithm.named_parameters()
                if parameter.requires_grad
            ],
        })
    return int(checkpoint.get("global_step", 0))

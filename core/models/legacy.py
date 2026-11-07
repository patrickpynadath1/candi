"""Convert legacy network configs and weights to ModularDiT.

The legacy classes (DiT, SimplexDiT, SphereDiT, ContinuousDiT) are presets of
ModularDiT. ``modular_config`` maps a legacy constructor config to the equivalent
ModularDiT config, and ``convert_state_dict`` renames a legacy state dict, so an
old checkpoint loads into the modular network unchanged. Used by the equivalence
tests and to regenerate earlier results from their checkpoints.

Legacy forward arguments map to the modular (states, points, time) interface as:
DiT / SimplexDiT ``conditioning`` and SphereDiT ``sigma`` -> global time with
``coordinate=time``; ContinuousDiT ``(time, sigma, alpha)`` -> global time plus
a point whose coefficients carry alpha and sigma.
"""

LEGACY_TARGETS = {
    "core.models.dit.DiT": "dit",
    "core.models.simplex.SimplexDiT": "simplex",
    "core.models.spherical.SphereDiT": "sphere",
    "core.models.continuous_dit.ContinuousDiT": "continuous",
}
HYBRID = "core.modifiers.hybrid.model.HybridModel"
MODULAR = "core.models.modular"
HYBRID_TIME = {
    "identity": "time",
    "negative_log_keep": "negative_log_keep",
    "log_one_minus_time": "log1p_negative_time",
}


def _slot(name, **options):
    return {"_target_": f"{MODULAR}.{name}", "_partial_": True, **options}


def _common(config, keys=("vocab_size", "hidden_size", "cond_dim", "n_blocks", "n_heads", "dropout")):
    return {k: config[k] for k in keys if k in config}


def hybrid_modular_config(config):
    """ModularDiT config for a legacy HybridModel (a DiT backbone over views)."""
    config = dict(config)
    backbone = dict(config["backbone"])
    if LEGACY_TARGETS.get(backbone.get("_target_")) != "dit":
        raise ValueError("only DiT-backed HybridModel configs are converted")
    if backbone.get("self_conditioning"):
        raise ValueError("a hybrid backbone's own self-conditioning is unused; remove it")
    result = modular_config({**backbone, "self_conditioning": False})
    result["input"] = _slot("ViewsInput", views=config["views"], fusion=config.get("fusion"))
    result["time"]["coordinate"] = HYBRID_TIME[config.get("time_conditioning", "identity")]
    result["output_vocab_size"] = config.get("output_vocab_size")
    if config.get("self_conditioning", False):
        result["self_conditioning"] = _slot(
            "SelfConditioning",
            source=config.get("self_conditioning_source", "posterior_mean"),
            initialization=config.get("conditioning_initialization", "zero"),
        )
    return result


def convert_hybrid_state_dict(state_dict, prefix=""):
    """Rename a legacy HybridModel state dict (under ``prefix``) for ModularDiT."""
    converted = {}
    backbone = prefix + "backbone."
    for key, value in state_dict.items():
        if not key.startswith(prefix):
            converted[key] = value
            continue
        rest = key[len(prefix):]
        if rest.startswith("backbone.vocab_embed."):
            continue  # The same tensor as the token view's table.
        if rest.startswith("backbone.feedback_table_adapter."):
            rest = "input.feedback_table." + rest[len("backbone.feedback_table_adapter."):]
        elif rest.startswith("backbone."):
            rest = next(iter(convert_state_dict({key: None}, "core.models.dit.DiT", backbone)))[len(backbone):]
        elif rest.startswith("views."):
            rest = "input." + rest
        elif rest.startswith("conditioning_adapter."):
            rest = "self_conditioning.adapter." + rest[len("conditioning_adapter."):]
        converted[prefix + rest] = value
    return converted


def modular_config(config):
    """ModularDiT config equivalent to a legacy network config (a plain dict)."""
    config = dict(config)
    if config.get("_target_") == HYBRID:
        return hybrid_modular_config(config)
    kind = LEGACY_TARGETS.get(config.pop("_target_", None))
    if kind is None:
        raise ValueError("not a legacy DiT/SimplexDiT/SphereDiT/ContinuousDiT config")
    for unsupported in ("input_adapter", "time_adapter", "output_adapter"):
        if config.get(unsupported) is not None:
            raise ValueError(f"legacy {unsupported} overrides are not converted")
    result = {"_target_": f"{MODULAR}.ModularDiT", **_common(config)}
    sc = config.get("self_conditioning", False)
    source = config.get("self_conditioning_source", "posterior_mean")
    if kind in {"dit", "simplex"}:
        adaLN = config.get("adaLN", False)
        result["input"] = (
            _slot("TokenInput")
            if kind == "dit"
            else _slot(
                "SimplexInput",
                mode=config.get("input_mode", "expectation"),
                precision=config.get("projection_precision", "legacy"),
                chunk_size=config.get("projection_chunk_size", 0),
            )
        )
        result["time"] = _slot("TimeConditioning", coordinate="time", adaLN=adaLN)
        result["output"] = _slot("AdaLNHead", adaLN=adaLN)
        result["block_adaLN"] = adaLN
        result["tie_word_embeddings"] = config.get("tie_word_embeddings", False)
        for key in ("sublayer_factory", "mlp_hidden_size", "block_execution"):
            if config.get(key) is not None:
                result[key] = config[key]
        if sc:
            result["self_conditioning"] = _slot("SelfConditioning", source=source)
        return result
    state_dim = config.get("state_dim")
    if kind == "sphere":
        state_dim = 512 if state_dim is None else state_dim
        result["state_dim"] = state_dim
        result["input"] = _slot("LinearInput", bias=True)
        result["time"] = _slot("TimeConditioning", coordinate="time")
        result["output"] = _slot(
            "AdaLNHead", temperature=config.get("learn_temperature_scaling", False)
        )
        if sc:
            result["self_conditioning"] = _slot("SelfConditioning", source=source, space="state")
        return result
    # ContinuousDiT
    state_dim = config["vocab_size"] if state_dim is None else state_dim
    result["state_dim"] = state_dim
    result["autocast_cache"] = False
    preconditioning = config.get("input_preconditioning", "gaussian")
    if config.get("input_architecture", "embedding_projection") == "linear":
        result["input"] = _slot(
            "LinearInput",
            bias=config.get("input_projection_bias", False),
            preconditioning=preconditioning,
        )
    else:
        result["input"] = _slot(
            "EmbeddedProjectionInput",
            input_dim=config.get("input_dim"),
            preconditioning=preconditioning,
        )
    result["time"] = _slot(
        "TimeConditioning",
        coordinate=config.get("time_conditioning", "log_one_minus_time"),
        epsilon=config.get("epsilon", 1e-6),
        input_projection=config.get("time_input", False),
    )
    result["output"] = (
        _slot("AdaLNHead")
        if config.get("output_architecture", "embedding_projection") == "adaln"
        else _slot("EmbeddingHead", output_dim=config.get("output_dim"))
    )
    if config.get("block_execution") is not None:
        result["block_execution"] = config["block_execution"]
    if sc:
        scale = config.get("conditioning_scale", 1.0)
        result["self_conditioning"] = _slot(
            "SelfConditioning",
            source=source,
            space="state",
            initialization=config.get("conditioning_initialization", "zero"),
            scale=state_dim**0.5 if scale == "unit_embedding" else float(scale),
        )
    return result


_RENAMES = {
    "dit": [("vocab_embed.", "input.embedding."), ("sigma_map.", "time.embedder."),
            ("output_layer.", "output.layer."), ("conditioning_adapter.", "self_conditioning.adapter.")],
    "sphere": [("input_projection.", "input.projection."), ("sigma_map.", "time.embedder."),
               ("output_layer.", "output.layer."), ("conditioning_adapter.", "self_conditioning.adapter.")],
    "continuous": [("input_embedding", "input.embedding"), ("input_projection.", "input.projection."),
                   ("sigma_map.", "time.embedder."), ("output_layer.", "output.layer."),
                   ("output_projection.", "output.projection."), ("output_embedding", "output.embedding"),
                   ("output_bias", "output.bias"), ("time_input_projection.", "time.input_projection."),
                   ("conditioning_adapter.", "self_conditioning.adapter.")],
}
_RENAMES["simplex"] = _RENAMES["dit"]


def convert_state_dict(state_dict, target, prefix=""):
    """Rename a legacy network's state dict keys (under ``prefix``) for ModularDiT."""
    kind = LEGACY_TARGETS[target]
    converted = {}
    for key, value in state_dict.items():
        if not key.startswith(prefix):
            converted[key] = value
            continue
        rest = key[len(prefix):]
        for old, new in _RENAMES[kind]:
            if rest.startswith(old):
                rest = new + rest[len(old):]
                break
        converted[prefix + rest] = value
    return converted


def convert_parameter_order(legacy_model, modular_model, values):
    """Reorder per-parameter values (e.g. EMA shadow weights) to ModularDiT order.

    EMA state is a list in ``parameters()`` order, which differs when a legacy
    class held bare parameters (ContinuousDiT's input/output embeddings).
    """
    target = next(t for t, k in LEGACY_TARGETS.items() if type(legacy_model).__name__ == t.rsplit(".", 1)[-1])
    names = [name for name, _ in legacy_model.named_parameters()]
    renamed = list(convert_state_dict(dict.fromkeys(names), target))
    by_name = dict(zip(renamed, values))
    return [by_name[name] for name, _ in modular_model.named_parameters()]

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
    del result["input"]
    result["views"] = config["views"]
    result["fusion"] = config.get("fusion")
    result["time"]["coordinate"] = HYBRID_TIME[config.get("time_conditioning", "identity")]
    result["output_vocab_size"] = config.get("output_vocab_size")
    result["self_conditioning"] = _slot(
        "SelfConditioning",
        enabled=bool(config.get("self_conditioning", False)),
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
        if rest.startswith("branches."):
            # Pre-modifier layout (as HybridModel's load hook remapped it).
            suffix = rest[len("branches."):]
            if ".adapter." in suffix:
                suffix = suffix.replace(".adapter.", ".representation.", 1)
            else:
                suffix = suffix.replace(".representation.", ".representation.state_representation.", 1)
            rest = "views." + suffix
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



def _legacy_config(legacy_model):
    """The constructor config of a built legacy network (blocks excluded)."""
    name = type(legacy_model).__name__
    common = {"vocab_size": legacy_model.vocab_size, "n_blocks": 0, "dropout": 0.0}
    common["cond_dim"] = legacy_model.sigma_map.mlp[0].out_features
    common["n_heads"] = legacy_model.blocks[0].n_heads
    if name == "SimplexDiT":
        adapter = legacy_model.input_adapter
        return {
            "_target_": "core.models.simplex.SimplexDiT",
            **common,
            "hidden_size": legacy_model.vocab_embed.embedding.shape[1],
            "adaLN": legacy_model.adaLN,
            "input_mode": legacy_model.input_mode,
            "projection_precision": getattr(adapter, "precision", "legacy"),
            "projection_chunk_size": getattr(adapter, "chunk_size", 0),
        }
    if name == "SphereDiT":
        return {
            "_target_": "core.models.spherical.SphereDiT",
            **common,
            "hidden_size": legacy_model.hidden_dim,
            "state_dim": legacy_model.state_dim,
            "learn_temperature_scaling": legacy_model.learn_temperature,
        }
    if name == "ContinuousDiT":
        linear = legacy_model.input_architecture == "linear"
        embedding_head = legacy_model.output_architecture == "embedding_projection"
        return {
            "_target_": "core.models.continuous_dit.ContinuousDiT",
            **common,
            "hidden_size": legacy_model.input_projection.out_features,
            "state_dim": legacy_model.state_dim,
            "input_dim": None if linear else legacy_model.input_embedding.shape[1],
            "output_dim": legacy_model.output_embedding.shape[1] if embedding_head else None,
            "epsilon": legacy_model.epsilon,
            "input_preconditioning": legacy_model.input_preconditioning,
            "time_conditioning": legacy_model.time_conditioning,
            "input_architecture": legacy_model.input_architecture,
            "output_architecture": legacy_model.output_architecture,
            "time_input": legacy_model.time_input_projection is not None,
            "input_projection_bias": linear and legacy_model.input_projection.bias is not None,
        }
    raise ValueError(f"no adopter for {name}")


def _move(model, path, value):
    *owners, leaf = path.split(".")
    owner = model
    for name in owners:
        owner = getattr(owner, name)
    setattr(owner, leaf, value)


def _adopt_state_network(legacy_model, time_coordinate=None):
    """SimplexDiT, SphereDiT or ContinuousDiT as its ModularDiT preset.

    The preset comes from ``modular_config`` (proven equal in
    test_modular_dit); the legacy submodules and parameters then replace the
    preset's under the ``_RENAMES`` table, so nothing is reinitialized.
    """
    import torch
    from hydra.utils import instantiate

    from . import modular

    config = _legacy_config(legacy_model)
    kind = LEGACY_TARGETS[config["_target_"]]
    preset = modular_config(config)
    if time_coordinate is not None:
        preset["time"]["coordinate"] = time_coordinate
    with torch.random.fork_rng(devices=[]):
        model = instantiate(preset)
    for old, new in _RENAMES[kind]:
        name = old.rstrip(".")
        if name in ("conditioning_adapter", "vocab_embed"):
            continue
        value = getattr(legacy_model, name, None)
        if value is not None:
            _move(model, new.rstrip("."), value)
    if kind == "simplex":
        model.input.embedding = legacy_model.vocab_embed
    model.block_execution = legacy_model.block_execution
    model.rotary_emb = legacy_model.rotary_emb
    model.blocks = legacy_model.blocks
    if getattr(model.output, "layer", None) is not None:
        model.output.apply_output = legacy_model.output_adapter
    adapter = legacy_model.conditioning_adapter
    if adapter is not None:
        model.self_conditioning = modular.SelfConditioning.wrap(
            adapter,
            space="hidden" if kind == "simplex" else "state",
            scale=getattr(legacy_model, "conditioning_scale", 1.0),
        )
    return model


def adopt(legacy_model, *, views=None, fusion=None, time_coordinate=None, fp32_clean_embedding=False):
    """Move a constructed legacy network into an equivalent ModularDiT.

    The legacy submodules (blocks, block policy, rotary, time embedder, output
    layer, self-conditioning adapter, token table) are reused as they are, so
    custom sublayers and depth mixing carry over, the parameters are the same
    tensors, and no random numbers are drawn.

    A token DiT with ``views`` binds its table to the first view's
    representation (as HybridModel did); without views it keeps a token input.
    SimplexDiT, SphereDiT and ContinuousDiT keep their single-state input slot:
    their recipes hold the view on the algorithm (``HybridDiffusion(views=...)``).
    ``time_coordinate`` overrides the network's own time coordinate.
    """
    import torch
    from functools import partial

    from . import modular
    from core.representations.noisy_inputs import TokenRepresentation

    if type(legacy_model).__name__ != "DiT":
        if views is not None or fusion is not None:
            raise ValueError("only the token DiT binds views; pass them to HybridDiffusion")
        return _adopt_state_network(legacy_model, time_coordinate)
    time_coordinate = "time" if time_coordinate is None else time_coordinate
    table = legacy_model.vocab_embed.embedding
    with torch.random.fork_rng(devices=[]):
        model = modular.ModularDiT(
            vocab_size=table.shape[0],
            hidden_size=table.shape[1],
            cond_dim=legacy_model.sigma_map.mlp[0].out_features,
            n_blocks=0,
            n_heads=legacy_model.blocks[0].n_heads,
            input=None if views is not None else partial(modular.TokenInput),
            views=views,
            fusion=fusion,
            time=partial(modular.TimeConditioning, coordinate=time_coordinate, adaLN=legacy_model.adaLN),
            output=partial(modular.AdaLNHead, adaLN=legacy_model.adaLN),
            autocast_cache=legacy_model.autocast_cache,
        )
    model.block_execution = legacy_model.block_execution
    model.rotary_emb = legacy_model.rotary_emb
    model.blocks = legacy_model.blocks
    model.time.embedder = legacy_model.sigma_map
    model.time_adapter = legacy_model.time_adapter
    model.output.layer = legacy_model.output_layer
    model.output.apply_output = legacy_model.output_adapter
    if views is None:
        model.input.embedding = legacy_model.vocab_embed
    else:
        view = model.input.views[0]
        if not isinstance(view.representation, TokenRepresentation):
            raise TypeError("the adopted token table needs a TokenRepresentation view")
        view.representation.embedding = table
        view.representation.fp32_clean_embedding = bool(fp32_clean_embedding)
    model.self_conditioning = (
        None
        if legacy_model.conditioning_adapter is None
        else modular.SelfConditioning.wrap(legacy_model.conditioning_adapter)
    )
    return model

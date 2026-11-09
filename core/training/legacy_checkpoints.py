"""Load checkpoints saved before REPLAiD became a recipe over shared components.

Continuous REPLAiD (``replaid``, ``replaid_sudoku_hard``) kept its state names:
``RePlaid`` was a GaussianDiffusion subclass, and the GaussianDiffusion recipe
registers the same modules. The REPLAiD hybrid's shared clock was split: its
endpoints are now the Gaussian view's LinearLogSNRSchedule parameters and its bin
logits the algorithm's PiecewiseImportanceTimeSampler. The old state dict also
held the shared clock under both views; the duplicate is dropped.

Old *configs* (``_target_: core.algorithms.replaid*.RePlaid*``) do not
instantiate any more. ``migrate_legacy_config`` rebuilds them from the current
preset, keeping the saved task, data, loader, trainer, evaluation and sampler
settings and the recipe values the old classes took as arguments; the runner
applies it to the run config and the loader to a checkpoint's saved config.
``migrate_legacy_checkpoint`` renames old hybrid state on load.
"""

import copy
import logging

# Removed family classes and the preset that now builds the same model.
LEGACY_RECIPES = {
    "core.algorithms.replaid.RePlaid": "replaid",
    "core.algorithms.replaid_hybrid.RePlaidHybrid": "replaid_hybrid",
}
# Sections the preset owns. The saved config keeps everything else (task, data,
# loader, trainer, evaluation, output paths, self_conditioning, token_space).
_PRESET_SECTIONS = (
    "algorithm",
    "model",
    "parameterization",
    "prediction",
    "representation",
    "process",
    "objective",
    "time_sampler",
    "noise_schedule",
)


def compose_preset(preset):
    """The repository's composed preset (unresolved), also inside a Hydra app.

    ``main.py`` holds an initialized GlobalHydra whose search path may be a
    bundle directory; set it aside while composing from ``configs/``.
    """
    from hydra.core.global_hydra import GlobalHydra
    from core.sweeps.configuration import compose

    instance = GlobalHydra.instance()
    active = instance.hydra if instance.is_initialized() else None
    instance.clear()
    try:
        return compose(preset)
    finally:
        if active is not None:
            GlobalHydra.instance().initialize(active)


def legacy_recipe(config):
    """The preset replacing a saved config's removed algorithm class, or None."""
    algorithm = config.get("algorithm") if config is not None else None
    target = algorithm.get("_target_") if algorithm is not None else None
    return LEGACY_RECIPES.get(target)


def _overlay(current, saved, where):
    """Saved values over the preset's, for the same component."""
    if current.get("_target_") != saved.get("_target_"):
        raise ValueError(
            f"cannot migrate {where}: saved {saved.get('_target_')} is not the "
            f"preset's {current.get('_target_')}"
        )
    for key, value in saved.items():
        if isinstance(value, dict) and isinstance(current.get(key), dict):
            _overlay(current[key], value, f"{where}.{key}")
        else:
            current[key] = copy.deepcopy(value)


def migrate_legacy_config_dict(saved, compose=None):
    """Rebuild a saved REPLAiD/REPLAiD-hybrid config (a plain dict) from its preset.

    Returns the dict unchanged when it names no removed class. ``compose`` maps a
    preset name to its composed, unresolved config (default: the repository's).
    """
    preset = legacy_recipe(saved)
    if preset is None:
        return saved
    compose = compose or compose_preset
    from omegaconf import OmegaConf

    current = OmegaConf.to_container(compose(preset), resolve=False)
    out = copy.deepcopy(saved)
    out.update({key: current[key] for key in _PRESET_SECTIONS})
    old = saved["algorithm"]
    _overlay(out["parameterization"], old["predictor"], "parameterization")
    _overlay(out["representation"], old["representation"], "representation")
    model = copy.deepcopy(old["model"])
    if preset == "replaid":
        _overlay(out["process"], old["process"], "process")
        _overlay(out["model"], model, "model")
        _overlay(out["noise_schedule"], old["schedule"], "noise_schedule")
        _overlay(out["time_sampler"], old["time_sampler"], "time_sampler")
        out["algorithm"]["corrupt_all"] = old.get("corrupt_all", False)
        if "selection" in old["prediction"]:
            out["prediction"]["selection"] = old["prediction"]["selection"]
    else:
        # The old class built the clean-context input itself around LinearInput.
        model_input = out["model"]["input"]
        _overlay(model_input["continuous"], model.pop("input"), "model.input")
        _overlay(out["model"], model, "model")
        model_input["context_scale"] = old.get("context_scale", "unit_embedding")
        clock = old["clock"]
        content = model_input["views"]["content"]
        content["schedule"].update(gamma_min=clock["lower"], gamma_max=clock["upper"])
        _overlay(content["process"], old["process"], "process")
        revealed = model_input["views"]["revealed"]["process"]
        _overlay(revealed["proposal"], old["corruption"], "corruption")
        out["time_sampler"].update(bins=clock["bins"], uniform_mix=clock["uniform_mix"])
    # Parameter names moved, so only the optimizer groups follow the preset.
    out["optimizer"] = {**saved.get("optimizer", {}), "groups": current["optimizer"]["groups"]}
    for path in (("sampler",), ("callback", "sampler_config")):
        section = out
        for key in path[:-1]:
            section = section.get(key) or {}
        sampler = section.get(path[-1])
        if sampler and sampler.get("_target_") == current["sampler"]["_target_"]:
            merged = copy.deepcopy(current["sampler"])
            _overlay(merged, sampler, ".".join(path))
            section[path[-1]] = merged
    return out


def migrate_legacy_config(config, compose=None):
    """``migrate_legacy_config_dict`` for an OmegaConf run config."""
    from omegaconf import OmegaConf

    preset = legacy_recipe(config)
    if preset is None:
        return config
    logging.getLogger(__name__).warning(
        "config names the removed %s; rebuilding it from the %s preset",
        config.algorithm._target_,
        preset,
    )
    saved = OmegaConf.to_container(config, resolve=False)
    return OmegaConf.create(migrate_legacy_config_dict(saved, compose))


def migrate_legacy_checkpoint(checkpoint, module):
    """Migrate a checkpoint saved by a removed REPLAiD class for ``module``.

    Rewrites the saved config (so the recipe check compares like with like) and,
    for the hybrid, renames the shared-clock state. Other checkpoints pass through.
    """
    hyper = checkpoint.get("hyper_parameters") or {}
    if legacy_recipe(hyper.get("config")) is not None:
        from omegaconf import OmegaConf

        # Saved configs are resolved; resolve the rebuilt one against itself.
        config = OmegaConf.create(migrate_legacy_config_dict(hyper["config"]))
        checkpoint = dict(checkpoint)
        checkpoint["hyper_parameters"] = {
            **hyper,
            "config": OmegaConf.to_container(config, resolve=True),
        }
    if any(".schedule.clock." in key for key in checkpoint["state_dict"]):
        checkpoint = migrate_replaid_hybrid_checkpoint(checkpoint, module)
    return checkpoint


_CLOCK = "model.input.views.{view}.schedule.clock."
_RENAMES = {
    "gamma_min": "model.input.views.0.schedule.gamma_min",
    "width_raw": "model.input.views.0.schedule.gamma_delta_raw",
    "clock_logits": "time_sampler.clock_logits",
}


def rename_replaid_key(key):
    """New name of an (optionally ``algorithm.``-prefixed) REPLAiD state key."""
    prefix = ""
    if key.startswith("algorithm."):
        prefix, key = "algorithm.", key[len("algorithm."):]
    for view in (0, 1):
        stem = _CLOCK.format(view=view)
        if key.startswith(stem) and key[len(stem):] in _RENAMES:
            return prefix + _RENAMES[key[len(stem):]]
    return prefix + key


def migrate_replaid_hybrid_state_dict(state_dict):
    migrated = {}
    for key, value in state_dict.items():
        new = rename_replaid_key(key)
        if new in migrated and key != new and ".views.1.schedule.clock." in key:
            continue  # the second view's alias of the shared clock
        migrated[new] = value
    return migrated


def _legacy_parameter_order(names):
    """The old parameter order: bin logits right after the clock endpoints."""
    logits = _RENAMES["clock_logits"]
    old = [name for name in names if name != logits]
    old.insert(old.index(_RENAMES["width_raw"]) + 1, logits)
    return old


def migrate_replaid_hybrid_checkpoint(checkpoint, module):
    """Rename state keys and reorder EMA shadows and optimizer state.

    ``module`` is the TrainingModule built from the current replaid_hybrid
    preset. The bin logits used to follow the clock endpoints in parameter
    order; they now come last (the time sampler is registered after the
    network), which also swaps their slot in the optimizer group they share
    with the embedding table.
    """
    algorithm = module.algorithm
    names = [name for name, _ in algorithm.named_parameters()]
    old = _legacy_parameter_order(names)
    checkpoint = dict(checkpoint)
    checkpoint["state_dict"] = migrate_replaid_hybrid_state_dict(checkpoint["state_dict"])
    ema = checkpoint.get("ema")
    if ema is not None:
        if len(ema["shadow_params"]) != len(old):
            raise ValueError("EMA state does not match this REPLAiD hybrid")
        shadow = dict(zip(old, ema["shadow_params"]))
        checkpoint["ema"] = {**ema, "shadow_params": [shadow[name] for name in names]}
    if checkpoint.get("optimizer_states"):
        ids = {id(parameter): name for name, parameter in algorithm.named_parameters()}
        groups = list(module.optimizer_parameters(module.config.optimizer))
        if groups and not isinstance(groups[0], dict):
            groups = [{"params": groups}]  # no configured groups
        groups = [[ids[id(parameter)] for parameter in group["params"]] for group in groups]
        rank = {name: index for index, name in enumerate(old)}
        before = [name for group in groups for name in sorted(group, key=rank.get)]
        after = {name: index for index, name in enumerate(name for group in groups for name in group)}
        checkpoint["optimizer_states"] = [
            {**state, "state": {after[before[index]]: value for index, value in state["state"].items()}}
            for state in checkpoint["optimizer_states"]
        ]
    return checkpoint

"""Convert checkpoints saved before the view-path unification.

No checkpoints are released, so core modules carry no load hooks for old
layouts. Conversion is explicit: ``convert_checkpoint`` maps an old Lightning
checkpoint onto today's ``TrainingModule`` for the same resolved run config
(``evaluation.legacy=convert``, ``scripts/convert_checkpoint.py``).

The key mapping is derived, not listed per family. An old standalone algorithm
stored each constructor argument under its own name (``model``, ``process``,
``schedule``, ...). ``conversion_plan`` instantiates those arguments once and
builds today's algorithm from those very objects; every tensor today's
algorithm holds is then one of the component tensors, so its old name is found
by identity (``adopt`` reuses the legacy network's tensors, views reuse the
process/schedule/representation buffers). A family that has not moved maps to
itself. Identity matching is used rather than value matching because seeded
initializations still collide (zero-initialized adapters, equal buffers).

Explicit name rules cover layouts no current constructor reproduces: the
pre-ModularDiT ``HybridModel`` (``backbone.*``, ``branches.*``; see
``convert_hybrid_state_dict``).

EMA shadow weights and optimizer state are stored by parameter position. Old
parameter order is today's, except within one component, where it follows the
old component's own order (e.g. ContinuousDiT's bare root parameters).
"""

from dataclasses import dataclass, field

import torch
from torch import nn

from core.models.legacy import convert_hybrid_state_dict


def _join(prefix, key):
    return f"{prefix}.{key}" if prefix else key


@dataclass
class ConversionPlan:
    """Old-layout names and positions for today's algorithm state.

    Keys are ``TrainingModule`` state keys (``algorithm.`` prefix).
    """

    algorithm: nn.Module
    names: dict = field(default_factory=dict)  # new key -> old key
    aliases: dict = field(default_factory=dict)  # old key -> all old keys of that tensor
    parameter_names: list = field(default_factory=list)  # trainable, today's order
    old_order: list = field(default_factory=list)  # parameter_names in old positional order
    hybrid_networks: list = field(default_factory=list)  # paths of view-input ModularDiTs

    def permutation(self):
        """``old_index[i]``: old position of today's i-th trainable parameter."""
        position = {name: i for i, name in enumerate(self.old_order)}
        return [position[name] for name in self.parameter_names]


def upgraded_config(config):
    """The config exactly as ``TrainingModule`` builds it."""
    from omegaconf import OmegaConf

    from core.modifiers.hybrid.checkpoint import upgrade_config

    raw = OmegaConf.to_container(config, resolve=False) if OmegaConf.is_config(config) else config
    return OmegaConf.create(upgrade_config(raw))


def _build_from_components(algorithm_config):
    """(algorithm, {argument: module}) with every argument instantiated once."""
    from hydra.utils import get_class, instantiate
    from omegaconf import OmegaConf

    if any(key in algorithm_config for key in ("_args_", "_partial_")):
        return None, {}
    kwargs, components = {}, {}
    for key in algorithm_config:
        if key.startswith("_"):
            continue
        value = algorithm_config[key]
        built = instantiate(value) if OmegaConf.is_config(value) else value
        kwargs[key] = built
        if isinstance(built, nn.Module):
            components[key] = built
    with torch.random.fork_rng(devices=[]):
        return get_class(algorithm_config._target_)(**kwargs), components


def _component_names(components):
    """Old names, keyed by tensor identity, and each component's parameter order."""
    by_tensor, order = {}, {}
    for name, component in components.items():
        for key, value in component.state_dict(keep_vars=True).items():
            by_tensor.setdefault(id(value), []).append(_join(name, key))
        order[name] = {
            _join(name, key): i for i, (key, _) in enumerate(component.named_parameters())
        }
    return by_tensor, order


def conversion_plan(config, algorithm=None):
    """Map today's algorithm state for ``config`` to the pre-unification layout.

    ``algorithm`` is today's built algorithm (default: built from ``config``).
    """
    from core.models.modular import ModularDiT
    from core.training.module import build_algorithm

    config = upgraded_config(config)
    if algorithm is None:
        with torch.random.fork_rng(devices=[]):
            algorithm = build_algorithm(config)
    plan = ConversionPlan(algorithm)
    modules = dict(algorithm.named_modules())
    plan.hybrid_networks = [
        path for path, module in modules.items()
        if isinstance(module, ModularDiT) and getattr(module, "views", None) is not None
    ]
    state = algorithm.state_dict(keep_vars=True)
    inner, components = _build_from_components(config.algorithm)
    renamed, owner = {}, {}  # full key -> old full key / (component, old name)
    if inner is not None:
        by_tensor, order = _component_names(components)
        inner_names = {}
        for key, value in inner.state_dict(keep_vars=True).items():
            old = by_tensor.get(id(value))
            if old is not None:
                inner_names[key] = old
        # Where the family algorithm sits (modifiers wrap it) and where each
        # of its components sits (encoder-decoder and block wrappers nest the
        # network). Old paths of the wrappers themselves are unchanged.
        family = [p for p, m in modules.items() if type(m) is type(inner)]
        if len(family) == 1:
            root = family[0]
            locations = {}
            for name, child in inner.named_children():
                keys = set(child.state_dict())
                candidates = [
                    p for p, m in modules.items()
                    if type(m) is type(child) and set(m.state_dict()) == keys
                ]
                direct = _join(root, name)
                locations[name] = direct if direct in candidates else (
                    candidates[0] if len(candidates) == 1 else direct
                )

            def place(name):
                child, _, rest = name.partition(".")
                return _join(locations[child], rest) if child in locations else _join(root, name)

            for key, old in inner_names.items():
                full = place(key)
                if full not in state:
                    continue
                renamed[full] = place(old[0])
                plan.aliases[renamed[full]] = [place(alias) for alias in old]
                component = old[0].partition(".")[0]
                owner[full] = (component, order[component].get(old[0]))
    for key in state:
        plan.names["algorithm." + key] = "algorithm." + renamed.get(key, key)
    plan.aliases = {
        "algorithm." + key: ["algorithm." + alias for alias in aliases]
        for key, aliases in plan.aliases.items()
    }
    trainable = [
        name for name, parameter in algorithm.named_parameters() if parameter.requires_grad
    ]
    plan.parameter_names = ["algorithm." + name for name in trainable]
    # Old order: today's slots, permuted only within each component.
    slots = {}
    for i, name in enumerate(trainable):
        component, index = owner.get(name, (None, None))
        if component is not None and index is not None:
            slots.setdefault(component, []).append((i, index, name))
    old_order = list(trainable)
    for entries in slots.values():
        positions = [i for i, _, _ in entries]
        for position, (_, _, name) in zip(positions, sorted(entries, key=lambda e: e[1])):
            old_order[position] = name
    plan.old_order = ["algorithm." + name for name in old_order]
    return plan


def _apply_hybrid_rules(state, plan):
    """Pre-ModularDiT HybridModel layouts under each view-input network."""
    for path in plan.hybrid_networks:
        prefix = "algorithm." + _join(path, "")
        if any(
            key.startswith(prefix + "backbone.") or key.startswith(prefix + "branches.")
            for key in state
        ):
            state = convert_hybrid_state_dict(state, prefix=prefix)
    return state


def convert_state_dict(state, plan):
    """Today's strict ``TrainingModule`` state from an old one.

    Every trainable parameter and every saved tensor must be used. Buffers the
    old checkpoint did not save (fixed by the config) keep their built values.
    """
    fresh = {"algorithm." + k: v for k, v in plan.algorithm.state_dict().items()}
    if set(state) == set(fresh):
        return dict(state), False
    state = _apply_hybrid_rules(dict(state), plan)
    trainable = {
        "algorithm." + key
        for key, value in plan.algorithm.state_dict(keep_vars=True).items()
        if isinstance(value, nn.Parameter) and value.requires_grad
    }
    converted, used = {}, set()
    for key, value in fresh.items():
        source = next((s for s in (plan.names.get(key, key), key) if s in state), None)
        if source is None:
            if key in trainable:
                raise ValueError(f"old checkpoint has no tensor for parameter {key!r}")
            converted[key] = value
            continue
        tensor = state[source]
        if tensor.shape != value.shape:
            raise ValueError(
                f"shape mismatch for {key}: checkpoint {tuple(tensor.shape)} (as {source}), "
                f"model {tuple(value.shape)}"
            )
        for alias in plan.aliases.get(source, [source]):
            if alias in state:
                if not torch.equal(state[alias], tensor):
                    raise ValueError(f"old checkpoint aliases differ: {source}, {alias}")
                used.add(alias)
        converted[key] = tensor
        used.add(source)
    unused = sorted(set(state) - used)
    if unused:
        raise ValueError(f"old checkpoint tensors have no place in this recipe: {unused[:10]}")
    return converted, True


def _reorder_ema(ema, plan):
    shadows = ema["shadow_params"]
    if len(shadows) != len(plan.parameter_names):
        raise ValueError(
            f"EMA holds {len(shadows)} tensors; the recipe has {len(plan.parameter_names)} parameters"
        )
    order = plan.permutation()
    return {**ema, "shadow_params": [shadows[i] for i in order]}, order != sorted(order)


def _reorder_optimizers(checkpoint, order):
    states = checkpoint.get("optimizer_states")
    if not states:
        return checkpoint, "unchanged"
    reordered = []
    for optimizer in states:
        groups = optimizer["param_groups"]
        if len(groups) != 1 or len(groups[0]["params"]) != len(order):
            checkpoint = {
                k: v for k, v in checkpoint.items() if k not in {"optimizer_states", "lr_schedulers"}
            }
            return checkpoint, "dropped (parameter groups cannot be reordered)"
        ids = groups[0]["params"]
        state = {
            ids[new]: optimizer["state"][ids[old]]
            for new, old in enumerate(order)
            if ids[old] in optimizer["state"]
        }
        reordered.append({**optimizer, "state": state})
    return {**checkpoint, "optimizer_states": reordered}, "reordered"


def convert_checkpoint(checkpoint, config, *, algorithm=None):
    """An old Lightning checkpoint that loads strictly into today's module.

    ``config`` is the run's resolved config; ``algorithm`` optionally reuses an
    already built ``TrainingModule.algorithm`` for it. The input is unchanged.
    """
    plan = conversion_plan(config, algorithm)
    state, renamed = convert_state_dict(checkpoint["state_dict"], plan)
    result = {**checkpoint, "state_dict": state, "core_format_version": 1}
    order = list(range(len(plan.parameter_names)))
    report = {"renamed": renamed, "ema": "absent", "optimizer": "unchanged"}
    if renamed and "ema" in checkpoint:
        result["ema"], permuted = _reorder_ema(checkpoint["ema"], plan)
        order = plan.permutation()
        report["ema"] = "reordered" if permuted else "unchanged"
    elif "ema" in checkpoint:
        report["ema"] = "unchanged"
    if order != sorted(order):
        result, report["optimizer"] = _reorder_optimizers(result, order)
    result["conversion"] = report
    return result

"""Learned time maps moved from the training time sampler slot to the information path.

Configs and checkpoints saved with the map as the time sampler (CDCD with
``adaptive_time_views``, the removed ``PiecewiseImportanceTimeSampler``, a simplex
adaptive sampler read by ``time_grid=adaptive``) must rebuild the same recipe and
sample the same tokens. The old layouts are written here from the documented
old configs (the real hybrid_sc / continuous_schedules / sc_vs_depth protocols
were also checked case by case when the move was made)."""

import copy

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from core.sweeps.configuration import compose
from core.tests.networks import TINY32
from core.training.checkpoints import load_weights
from core.training.legacy_checkpoints import (
    migrate_legacy_config,
    migrate_time_map_config_dict,
    migrate_time_map_state_dict,
)
from core.training.module import TrainingModule

CDCD_SAMPLER = {
    "_target_": "core.time_sampling.cdcd.CDCDTimeSampler", "adaptive": True, "bins": 100,
    "min_bin_size": 0.0001, "learning_rate": 0.001, "momentum": 0.99, "warmup": 2,
    "refit_every": 1, "epsilon": 0.001, "antithetic": True,
}
PIECEWISE_SAMPLER = {
    "_target_": "core.time_sampling.importance.PiecewiseImportanceTimeSampler",
    "bins": 64, "uniform_mix": 0.01,
}
LOSS_DERIVATIVE_SAMPLER = {
    "_target_": "core.time_sampling.loss_derivative.LossDerivativeTimeSampler",
    "adaptive": True, "buffer_size": 64, "bins": 50, "grid_size": 1000, "warmup": 2,
    "refit_every": 1, "momentum": 0.9, "epsilon": 1e-05, "antithetic": True,
}
PLAIN_SIMPLEX_SAMPLER = {  # configs/experiment/simplex.yaml's former time sampler
    **LOSS_DERIVATIVE_SAMPLER, "adaptive": False, "buffer_size": 65536, "warmup": 1000,
    "refit_every": 50,
}
RECIPES = {
    "simplex": ("sudoku_simplex", [
        "model.hidden_size=16", "model.n_heads=2", "model.n_blocks=1", "model.cond_dim=8",
        "model.dropout=0", "algorithm.length=6", "sampler.steps=3"], PLAIN_SIMPLEX_SAMPLER),
    "cdcd_views": ("corruption_factorial", [
        *TINY32, "+ablation_path=gaussian", "+information_path/discrete_view=masked",
        "information_path/path_schedule=cdcd", "task.vocab_size=7", "task.length=6",
        "sampler.steps=3", "time_map.warmup=2"], CDCD_SAMPLER),
    "replaid_hybrid": ("replaid_hybrid", [
        *TINY32, "task.vocab_size=5", "task.length=6", "sampler.steps=3"], PIECEWISE_SAMPLER),
    "simplex_adaptive_grid": ("sudoku_simplex_adaptive", [
        "model.hidden_size=16", "model.n_heads=2", "model.n_blocks=1", "model.cond_dim=8",
        "model.dropout=0", "algorithm.length=6", "sampler.time_grid=adaptive", "sampler.steps=3",
        "time_map.warmup=2", "time_map.refit_every=1", "time_map.buffer_size=64"],
        LOSS_DERIVATIVE_SAMPLER),
}


def new_config(name, extra=()):
    preset, overrides, _ = RECIPES[name]
    return compose(preset, [*overrides, "training.ema=0.9", *extra])


def old_layout(config, sampler):
    """The same recipe in the old layout: the map is the training time sampler."""
    old = OmegaConf.to_container(config, resolve=False)
    old["time_sampler"] = copy.deepcopy(sampler)
    del old["time_map"], old["algorithm"]["time_map"]
    for group in old.get("optimizer", {}).get("groups") or []:
        group["patterns"] = [p.replace("time_map.", "time_sampler.", 1) for p in group["patterns"]]
    return old


def old_state(state):
    return {key.replace(".time_map.", ".time_sampler."): value for key, value in state.items()}


@pytest.mark.parametrize("name", sorted(RECIPES))
@pytest.mark.parametrize("resolved", [False, True])
def test_old_configs_migrate_to_the_new_preset(name, resolved):
    config = new_config(name)
    old = old_layout(config, RECIPES[name][2])
    expected = OmegaConf.to_container(config, resolve=resolved)
    if resolved:
        old = OmegaConf.to_container(OmegaConf.create(old), resolve=True)
    migrated = migrate_time_map_config_dict(old)
    for key in ("algorithm", "time_sampler", "time_map", "optimizer"):
        assert migrated.get(key) == expected.get(key), key
    assert migrate_time_map_config_dict(migrated) is migrated  # idempotent
    assert migrate_time_map_config_dict(expected) is expected  # new layouts pass through


def test_loss_derivative_always_migrates_and_cdcd_only_where_generation_reads_it():
    config = new_config("simplex_adaptive_grid", ["time_map.adaptive=false", "sampler.time_grid=linear"])
    old = old_layout(config, {**LOSS_DERIVATIVE_SAMPLER, "adaptive": False})
    # The loss-derivative clock is a learned warp: even switched off (the simplex
    # preset) and on a linear grid it is the time map, never a training sampler.
    migrated = migrate_time_map_config_dict(old)
    assert migrated["time_map"] == OmegaConf.to_container(config.time_map)
    assert migrated["time_sampler"] == OmegaConf.to_container(config.time_sampler)
    cdcd = {**CDCD_SAMPLER, "epsilon": 1e-05}
    old = old_layout(config, cdcd)
    # A training-only adaptive CDCD sampler stays a training time sampler ...
    assert migrate_time_map_config_dict(old) is old
    # ... unless generation reads it or the module it loads into has a map.
    old["sampler"]["time_grid"] = "adaptive"
    assert migrate_time_map_config_dict(old)["algorithm"]["time_map"] == "${time_map}"
    old["sampler"]["time_grid"] = "linear"
    assert migrate_time_map_config_dict(old, force=True)["algorithm"]["time_map"] == "${time_map}"


def test_adaptive_views_require_a_time_map():
    config = OmegaConf.create(old_layout(new_config("cdcd_views"), CDCD_SAMPLER))
    with pytest.raises(Exception, match="algorithm.time_map"):  # Hydra wraps the TypeError
        instantiate(config.algorithm)
    # The runner migrates such a run config before building it.
    algorithm = instantiate(migrate_legacy_config(config).algorithm)
    assert type(algorithm.time_map).__name__ == "CDCDTimeMap"


def train_and_sample(module, config, steps=4):
    algorithm = module.algorithm
    torch.manual_seed(5)
    tokens = torch.randint(1, int(config.task.vocab_size) - 1, (4, algorithm.length))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, :2] = False
    optimizer = torch.optim.AdamW(module.optimizer_parameters(config.optimizer), lr=1e-2)
    for step in range(steps):
        optimizer.zero_grad()
        algorithm.loss(tokens, mask, training=True, step=step)["loss"].backward()
        optimizer.step()
        module.ema.update(algorithm.parameters())
        for component in algorithm.modules():
            hook = getattr(component, "post_optimizer_step", None)
            if callable(hook):
                hook()
    return tokens, mask


def sample(module, config, tokens, mask):
    module.algorithm.eval()
    torch.manual_seed(12)
    return module.algorithm.generate(instantiate(config.sampler), tokens, ~mask)[0]


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_old_format_checkpoint_loads_and_samples_identically(name, tmp_path):
    config = new_config(name)
    module = TrainingModule(config)
    tokens, mask = train_and_sample(module, config)
    # The loss-CDF maps are fitted (not the identity) before sampling.
    time_map = module.algorithm.time_map
    assert bool(getattr(time_map, "ready", True)) or not getattr(time_map, "adaptive", True)
    expected = sample(module, config, tokens, mask)
    old_config = OmegaConf.to_container(
        OmegaConf.create(old_layout(config, RECIPES[name][2])), resolve=True
    )
    if name == "simplex_adaptive_grid":
        old_config["sampler"]["time_grid"] = "linear"  # saved by training (callback grid)
    path = tmp_path / "old.ckpt"
    torch.save({
        "core_format_version": 1, "global_step": 4,
        "state_dict": old_state(module.state_dict()), "ema": module.ema.state_dict(),
        "hyper_parameters": {"config": old_config},
    }, path)
    torch.manual_seed(0)
    fresh = TrainingModule(config)
    load_weights(fresh, path)  # also runs the recipe check against the migrated config
    for key, value in module.state_dict().items():
        assert torch.equal(fresh.state_dict()[key], value), key
    assert torch.equal(sample(fresh, config, tokens, mask), expected)


def test_resume_renames_old_time_sampler_state():
    config = new_config("replaid_hybrid")
    module = TrainingModule(config)
    state = old_state(module.state_dict())
    assert "algorithm.time_sampler.clock_logits" in state
    checkpoint = {"state_dict": state, "ema": module.ema.state_dict()}
    module.on_load_checkpoint(checkpoint)  # Lightning loads the state dict after this hook
    assert checkpoint["state_dict"].keys() == module.state_dict().keys()


def test_state_rename_touches_only_keys_the_module_expects():
    expected = {"algorithm.time_map.ready", "algorithm.model.time_sampler.x"}
    renamed = migrate_time_map_state_dict(
        {"algorithm.time_sampler.ready": 1, "algorithm.model.time_sampler.x": 2,
         "algorithm.time_sampler.other": 3},
        expected,
    )
    assert renamed == {"algorithm.time_map.ready": 1, "algorithm.model.time_sampler.x": 2,
                       "algorithm.time_sampler.other": 3}

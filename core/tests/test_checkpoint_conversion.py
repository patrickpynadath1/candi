"""Checkpoints saved before the view-path unification convert and load strictly.

An old checkpoint is made by a ``TrainingModule`` whose algorithm is the frozen
standalone class (core/tests/reference_algorithms), with perturbed weights and
EMA so that nothing can pass by coincidence of initialization. The converted
checkpoint must load strictly into today's module for the same config and give
the same parameters, buffers, EMA, and samples (same RNG draw order).
"""

from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

from core.models.legacy_checkpoints import conversion_plan, convert_checkpoint
from core.sweeps import compose
from core.tests.view_equivalence import SMALL, assert_sampling_identical, batch
from core.training.checkpoints import load_weights
from core.training.module import TrainingModule

MDLM = "core.tests.reference_algorithms.mdlm.MDLM"
UNIFORM = "core.tests.reference_algorithms.uniform_posterior.PosteriorUniformDiffusion"

MDLM_SAMPLERS = [
    {"_target_": "core.samplers.ancestral.AncestralSampler", "steps": 8, "final_denoise": "greedy"},
    {"_target_": "core.samplers.ancestral.AncestralSampler", "steps": 8, "final_denoise": "ancestral"},
    {"_target_": "core.samplers.entropy_bounded.EntropyBoundedSampler", "entropy_bound": 0.3, "max_steps": 180},
]
UNIFORM_SAMPLERS = [
    {"_target_": "core.samplers.uniform_ancestral.UniformAncestralSampler", "steps": 8,
     "epsilon": 1e-3, "final_denoise": final}
    for final in ("ancestral", "greedy")
]
CASES = {
    "mdlm": ("sudoku_hard", MDLM, {}, MDLM_SAMPLERS),
    "mdlm_adaln_sc": (
        "sudoku_hard", MDLM,
        {"model.adaLN": True, "algorithm.time_conditioning": True, "self_conditioning.enabled": True},
        MDLM_SAMPLERS,
    ),
    "uniform_posterior": ("uniform_posterior", UNIFORM, {}, UNIFORM_SAMPLERS),
    "uniform_posterior_sc": ("uniform_posterior", UNIFORM, {"self_conditioning.enabled": True}, UNIFORM_SAMPLERS),
}


def config(experiment, overrides, target=None):
    cfg = compose(experiment, {**SMALL, "training.ema": 0.9, **overrides})
    if target is not None:
        OmegaConf.update(cfg, "algorithm._target_", target)
    return cfg


def perturb(module, seed=5):
    """Trained-looking weights and an EMA that differs from them."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.algorithm.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator))
    module.ema.shadow_params = [
        p.detach() + torch.randn(p.shape, generator=generator) for p in module.algorithm.parameters()
    ]
    module.ema.num_updates = 17


def saved(module, step=3):
    checkpoint = {"state_dict": module.state_dict(), "global_step": step}
    module.on_save_checkpoint(checkpoint)
    return checkpoint


def old_checkpoint(name):
    experiment, reference, overrides, _ = CASES[name]
    torch.manual_seed(0)
    old = TrainingModule(config(experiment, overrides, reference))
    perturb(old)
    return old, saved(old)


def assert_same_module(old, new):
    old_parameters, new_parameters = list(old.algorithm.parameters()), list(new.algorithm.parameters())
    assert len(old_parameters) == len(new_parameters)
    for a, b in zip(old_parameters, new_parameters):
        assert torch.equal(a, b)
    assert len(old.ema.shadow_params) == len(new.ema.shadow_params)
    for a, b in zip(old.ema.shadow_params, new.ema.shadow_params):
        assert torch.equal(a, b)
    assert new.ema.num_updates == old.ema.num_updates
    old_buffers = sorted((tuple(b.shape), b.sum().item()) for b in old.algorithm.buffers())
    new_buffers = sorted((tuple(b.shape), b.sum().item()) for b in new.algorithm.buffers())
    assert old_buffers == new_buffers


def assert_same_samples(name, old, new):
    experiment, _, overrides, samplers = CASES[name]
    tokens, _ = batch(config(experiment, overrides))
    prompt = tokens[:3]
    fixed = torch.zeros_like(prompt, dtype=torch.bool)
    fixed[:, :6] = True
    assert_sampling_identical(old.algorithm, new.algorithm, prompt, fixed, samplers)


@pytest.mark.parametrize("name", sorted(CASES))
def test_old_core_checkpoint_converts_loads_and_samples_identically(name, tmp_path):
    experiment, _, overrides, _ = CASES[name]
    old, checkpoint = old_checkpoint(name)
    keys = set(checkpoint["state_dict"])
    torch.manual_seed(123)
    new = TrainingModule(config(experiment, overrides))
    with pytest.raises(RuntimeError):
        new.load_state_dict(checkpoint["state_dict"], strict=True)
    converted = convert_checkpoint(checkpoint, new.config)
    assert set(checkpoint["state_dict"]) == keys  # The input is not modified.
    assert converted["conversion"]["renamed"]
    new.load_state_dict(converted["state_dict"], strict=True)
    new.ema.load_state_dict(converted["ema"])
    assert_same_module(old, new)
    assert_same_samples(name, old, new)
    # The same through evaluation.legacy=convert.
    path = tmp_path / "old.ckpt"
    torch.save(checkpoint, path)
    torch.manual_seed(321)
    loaded = TrainingModule(config(experiment, overrides))
    assert load_weights(loaded, path, legacy="convert") == 3
    assert_same_module(old, loaded)
    with pytest.raises(RuntimeError):
        load_weights(TrainingModule(config(experiment, overrides)), path)


def test_current_checkpoint_converts_as_identity():
    module = TrainingModule(config("sudoku_hard", {}))
    perturb(module)
    checkpoint = saved(module)
    converted = convert_checkpoint(checkpoint, module.config)
    assert not converted["conversion"]["renamed"]
    assert converted["state_dict"].keys() == checkpoint["state_dict"].keys()
    assert all(converted["ema"]["shadow_params"][i] is s for i, s in enumerate(checkpoint["ema"]["shadow_params"]))


def test_unmigrated_family_maps_to_itself():
    cfg = config("sedd_absorb", {})
    plan = conversion_plan(cfg)
    assert all(new == old for new, old in plan.names.items())
    assert plan.old_order == plan.parameter_names


@pytest.mark.parametrize("experiment", ["bd3lm", "e2d2"])
def test_wrapped_mdlm_networks_are_renamed_in_place(experiment):
    overrides = {"model.n_blocks": 2}
    if experiment == "e2d2":
        overrides["encoder_decoder.encoder_layers"] = 1
    torch.manual_seed(0)
    old = TrainingModule(config(experiment, overrides, MDLM))
    perturb(old)
    checkpoint = saved(old)
    torch.manual_seed(9)
    new = TrainingModule(config(experiment, overrides))
    converted = convert_checkpoint(checkpoint, new.config, algorithm=new.algorithm)
    assert any(".vocab_embed." in key for key in checkpoint["state_dict"])
    new.load_state_dict(converted["state_dict"], strict=True)
    new.ema.load_state_dict(converted["ema"])
    assert_same_module(old, new)


def test_published_mdlm_import_goes_through_the_converter(tmp_path):
    old, _ = old_checkpoint("mdlm")
    published = {
        "state_dict": {
            "backbone." + key.removeprefix("algorithm.model."): value
            for key, value in old.state_dict().items()
            if key.startswith("algorithm.model.")
        },
        "ema": old.ema.state_dict(),
        "global_step": 9,
    }
    path = tmp_path / "published.ckpt"
    torch.save(published, path)
    torch.manual_seed(77)
    new = TrainingModule(config("sudoku_hard", {}))
    assert load_weights(new, path, legacy=True) == 9
    assert_same_module(old, new)
    assert_same_samples("mdlm", old, new)


def test_missing_and_unused_tensors_fail_explicitly():
    _, checkpoint = old_checkpoint("mdlm")
    cfg = config("sudoku_hard", {})
    state = dict(checkpoint["state_dict"])
    del state["algorithm.model.output_layer.linear.weight"]
    with pytest.raises(ValueError, match="no tensor for parameter"):
        convert_checkpoint({**checkpoint, "state_dict": state}, cfg)
    state = {**checkpoint["state_dict"], "algorithm.model.stray": torch.zeros(1)}
    with pytest.raises(ValueError, match="no place"):
        convert_checkpoint({**checkpoint, "state_dict": state}, cfg)


def test_optimizer_state_follows_the_ema_permutation():
    from core.models.legacy_checkpoints import _reorder_optimizers

    checkpoint = {
        "optimizer_states": [{
            "state": {0: {"m": 0}, 1: {"m": 1}, 2: {"m": 2}},
            "param_groups": [{"params": [0, 1, 2], "lr": 1}],
        }],
    }
    result, status = _reorder_optimizers(checkpoint, [2, 0, 1])
    assert status == "reordered"
    assert [result["optimizer_states"][0]["state"][i]["m"] for i in range(3)] == [2, 0, 1]
    grouped = {"optimizer_states": [{"state": {}, "param_groups": [{"params": [0]}, {"params": [1, 2]}]}],
               "lr_schedulers": [{}]}
    result, status = _reorder_optimizers(grouped, [2, 0, 1])
    assert status.startswith("dropped") and "optimizer_states" not in result


def test_convert_checkpoint_script_writes_a_new_file(tmp_path, monkeypatch):
    import importlib.util
    import sys

    old, checkpoint = old_checkpoint("uniform_posterior")
    source, target, run_config = tmp_path / "old.ckpt", tmp_path / "new.ckpt", tmp_path / "config.yaml"
    torch.save(checkpoint, source)
    experiment, _, overrides, _ = CASES["uniform_posterior"]
    OmegaConf.save(config(experiment, overrides), run_config, resolve=True)
    spec = importlib.util.spec_from_file_location(
        "convert_checkpoint", Path(__file__).resolve().parents[2] / "scripts/convert_checkpoint.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    arguments = ["convert_checkpoint.py", str(source), str(target), "--config", str(run_config)]
    monkeypatch.setattr(sys, "argv", arguments)
    script.main()
    new = TrainingModule(OmegaConf.load(run_config))
    assert load_weights(new, target) == 3
    assert_same_module(old, new)
    before = source.read_bytes()
    with pytest.raises(FileExistsError):
        script.main()
    monkeypatch.setattr(sys, "argv", [*arguments[:2], str(source), *arguments[3:]])
    with pytest.raises(FileExistsError):
        script.main()
    assert source.read_bytes() == before


def test_warm_start_converts_only_when_asked(tmp_path):
    old, checkpoint = old_checkpoint("mdlm")
    path = tmp_path / "old.ckpt"
    torch.save(checkpoint, path)
    cfg = config("sudoku_hard", {})
    OmegaConf.update(cfg, "mode", "train")
    OmegaConf.update(cfg, "training.initialize_from", str(path), force_add=True)
    with pytest.raises(RuntimeError):
        TrainingModule(cfg)
    OmegaConf.update(cfg, "training.initialize_convert", True, force_add=True)
    new = TrainingModule(cfg)
    for a, b in zip(old.ema.shadow_params, new.algorithm.parameters()):
        assert torch.equal(a, b)  # Warm start copies the EMA weights in.


class StandaloneToy(torch.nn.Module):
    """An old standalone family: it stored its network as ``model``."""

    def __init__(self, model, scale):
        super().__init__()
        self.model, self.scale = model, scale


class AdoptedToy(torch.nn.Module):
    """The same family moved onto ModularDiT (another parameter order)."""

    def __init__(self, model, scale):
        from core.models.legacy import adopt

        super().__init__()
        self.model, self.scale = adopt(model), scale


def test_ema_and_optimizer_follow_a_changed_parameter_order():
    from hydra.utils import instantiate

    network = {"_target_": "core.models.continuous_dit.ContinuousDiT", "vocab_size": 7, "hidden_size": 16,
               "cond_dim": 8, "n_blocks": 1, "n_heads": 2, "dropout": 0.0}
    scale = {"_target_": "torch.nn.Linear", "in_features": 2, "out_features": 3}
    old_config = OmegaConf.create({"algorithm": {"_target_": f"{__name__}.StandaloneToy", "model": network, "scale": scale}})
    new_config = OmegaConf.create({"algorithm": {"_target_": f"{__name__}.AdoptedToy", "model": network, "scale": scale}})
    torch.manual_seed(0)
    old = instantiate(old_config.algorithm)
    with torch.no_grad():
        for parameter in old.parameters():
            parameter.add_(torch.randn_like(parameter))
    new = instantiate(new_config.algorithm)
    shadows = [torch.full(p.shape, float(i)) for i, p in enumerate(old.parameters())]
    optimizer = {"state": {i: {"step": i} for i in range(len(shadows))},
                 "param_groups": [{"params": list(range(len(shadows)))}]}
    checkpoint = {
        "state_dict": {"algorithm." + k: v for k, v in old.state_dict().items()},
        "ema": {"decay": 0.9, "num_updates": 1, "shadow_params": shadows},
        "optimizer_states": [optimizer],
    }
    converted = convert_checkpoint(checkpoint, new_config, algorithm=new)
    assert converted["conversion"] == {"renamed": True, "ema": "reordered", "optimizer": "reordered"}
    new.load_state_dict({k.removeprefix("algorithm."): v for k, v in converted["state_dict"].items()})
    old_parameters = list(old.parameters())
    for position, (parameter, shadow) in enumerate(zip(new.parameters(), converted["ema"]["shadow_params"])):
        index = next(i for i, p in enumerate(old_parameters) if p.shape == parameter.shape and torch.equal(p, parameter))
        assert torch.equal(shadow, shadows[index])
        assert converted["optimizer_states"][0]["state"][position]["step"] == index

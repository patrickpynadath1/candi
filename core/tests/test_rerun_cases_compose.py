"""Every rerun case composes for every policy's starting point, as the evaluate worker does.

For all 99 cases of the three rerun manifests and every inference policy, applies the
evaluation overrides, the policy's sampler and its first initial point (or a random
draw from its space) with ``nested_tuning.evaluation_config``, checks the recipe is
unchanged apart from the tuned values, and instantiates the algorithm (at test size)
and the sampler. With ``SDDC_RERUN_CHECKPOINTS="<case>=<last.ckpt>,..."`` it also
loads those source checkpoints at full size, with the recipe check on, and generates.
"""

import json
import os

import optuna
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from core.sweeps import nested_tuning
from core.tests.helpers import ROOT, run_python
from core.tests.networks import TINY16_SIZES

SWEEPS = ("sc_vs_depth", "continuous_schedules", "hybrid_sc")


@pytest.fixture(scope="module")
def manifests(tmp_path_factory):
    root = tmp_path_factory.mktemp("rerun")
    env = {**os.environ, "PYTHONPATH": str(ROOT), "SDDC_STORAGE_ROOT": str(root),
           "SDDC_SOURCE_RUNS": str(root / "h100")}
    result = {}
    for sweep in SWEEPS:
        plan = root / sweep
        run_python("-m", "core.sweeps.prepare", "-m", f"+sweep={sweep}_rerun",
                   f"preparation.output={plan}", env=env)
        run_python("-m", "core.sweeps.prepare", "collect", str(plan), env=env)
        result[sweep] = json.loads((plan / "manifest.json").read_text())
    return result


def starting_point(policy):
    points = policy.get("initial_points")
    if points:
        return points[0]
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    return nested_tuning.suggest(study.ask(), policy["space"])


def tuning_evaluation(case):
    hidden = ("test_indices", "test_seeds", "tuning_indices", "test_score", "test_overrides")
    evaluation = {k: v for k, v in case["evaluation"].items() if k not in hidden}
    return {**evaluation, "indices": case["evaluation"]["tuning_indices"][:2]}


def flat(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flat(child, f"{prefix}.{key}" if prefix else key)
    else:
        yield prefix, value


@pytest.mark.parametrize("sweep", SWEEPS)
def test_every_policy_composes_and_instantiates_for_every_case(manifests, sweep):
    cases = manifests[sweep]
    assert len(cases) == {"sc_vs_depth": 27, "continuous_schedules": 24, "hybrid_sc": 48}[sweep]
    for case in cases:
        base = OmegaConf.load(case["config"])
        recipe = dict(flat(OmegaConf.to_container(base.algorithm, resolve=True)))
        for policy in case["inference"]:
            point = starting_point(policy)
            config = nested_tuning.evaluation_config(
                OmegaConf.load(case["config"]), tuning_evaluation(case), policy, point
            )
            # The tuned values land on the real recipe nodes; nothing else changes.
            tuned = dict(flat(OmegaConf.to_container(config.algorithm, resolve=True)))
            expected = dict(recipe)
            for key, value in point.items():
                if key.startswith("algorithm."):
                    assert key.removeprefix("algorithm.") in recipe, (case["name"], key)
                    expected[key.removeprefix("algorithm.")] = value
            changed = sorted(k for k in set(tuned) | set(expected)
                             if tuned.get(k, "<absent>") != expected.get(k, "<absent>"))
            assert not changed, (case["name"], policy["name"], changed[:5])
            for key, value in TINY16_SIZES.items():
                OmegaConf.update(config, key, value)
            try:
                instantiate(config.algorithm)
                instantiate(config.sampler)
            except Exception as error:  # report the case and policy, not just the error
                raise AssertionError(f"{case['name']} / {policy['name']}: {error}") from error


def checkpoints():
    pairs = [p for p in os.environ.get("SDDC_RERUN_CHECKPOINTS", "").split(",") if "=" in p]
    return [tuple(p.split("=", 1)) for p in pairs]


@pytest.mark.skipif(not checkpoints(), reason="set SDDC_RERUN_CHECKPOINTS=<case>=<last.ckpt>,...")
@pytest.mark.parametrize("name,path", checkpoints() or [("", "")])
def test_source_checkpoint_loads_and_generates_under_every_policy(manifests, name, path):
    from core.training.checkpoints import load_weights
    from core.training.module import TrainingModule

    case = next(c for cases in manifests.values() for c in cases if c["name"].startswith(name))
    for policy in case["inference"]:
        config = nested_tuning.evaluation_config(
            OmegaConf.load(case["config"]), {**tuning_evaluation(case), "nfe": 4},
            policy, starting_point(policy),
        )
        config.training.performance.compile.enabled = False
        module = TrainingModule(config)
        load_weights(module, path)  # strict weights and the recipe check
        length = module.algorithm.length
        tokens = torch.zeros(2, length, dtype=torch.long)
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        with module.evaluation_weights(True):
            output, info = module.algorithm.generate(instantiate(config.sampler), tokens, mask)
        assert output.shape == tokens.shape and max(info["nfe_per_sample"]) <= 4, policy["name"]


def test_updates_write_through_interpolated_parents():
    config = OmegaConf.create({
        "parameterization": {"_target_": "x.Posterior", "temperature": 1.0, "top_p": 1.0},
        "algorithm": {"predictor": "${parameterization}", "mode": "a"},
        "other": {"uses": "${parameterization.temperature}"},
    })
    nested_tuning.update(config, "algorithm.predictor.temperature", 0.7, merge=False, force_add=True)
    nested_tuning.update(config, "algorithm.mode", "b", merge=False, force_add=True)
    assert OmegaConf.is_interpolation(config.algorithm, "predictor")  # the alias survives
    assert config.algorithm.predictor._target_ == "x.Posterior"
    assert config.parameterization.temperature == 0.7 == config.other.uses
    assert config.algorithm.mode == "b"
    assert nested_tuning.resolve_key(config, "task.sol.bootstrap") == "task.sol.bootstrap"
    odd = OmegaConf.create({"a": {"b": "prefix-${x}"}, "x": 1})
    with pytest.raises(ValueError, match="interpolation"):
        nested_tuning.resolve_key(odd, "a.b.c")


def test_recipe_check_accepts_only_default_values_of_keys_added_later():
    from core.training import checkpoints

    saved = {"algorithm": {"prediction": {"enabled": True}, "length": 8}}
    current = {"algorithm": {"prediction": {"enabled": True, "selected_gradients": "all",
                                            "fixed_feedback": "prediction"}, "length": 8}}
    assert checkpoints.recipe_mismatches(saved, current) == []
    changed = json.loads(json.dumps(current))
    changed["algorithm"]["prediction"]["fixed_feedback"] = "clean"
    assert checkpoints.recipe_mismatches(saved, changed) == ["algorithm.prediction.fixed_feedback"]
    other = json.loads(json.dumps(current))
    other["algorithm"]["new_option"] = 1  # not a registered addition: still a mismatch
    assert checkpoints.recipe_mismatches(saved, other) == ["algorithm.new_option"]
    # A key the checkpoint has but the current recipe lacks is still a mismatch.
    removed = {"algorithm": {"prediction": {"enabled": True, "selected_gradients": "all"},
                             "length": 8}}
    assert checkpoints.recipe_mismatches(removed, saved) == ["algorithm.prediction.selected_gradients"]

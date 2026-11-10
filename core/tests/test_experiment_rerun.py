"""experiments/rerun: inference-only reruns of sc-vs-depth, continuous-schedules, hybrid-sc.

Checks the evaluation partitions, prepares the three rerun sweeps next to their source
sweeps (the cases must reuse the source recipes and checkpoints unchanged), and runs
every new policy's sampler on a tiny recipe of each corruption family.
"""

import hashlib
import json
import os

import optuna
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from core.sweeps import nested_tuning
from core.tests.helpers import ROOT, run_python
from core.tests.test_sampler_equivalence import build_algorithm

EVALUATION = ROOT / "experiments/rerun/evaluation"
CANDIDATE = hashlib.sha256(json.dumps({}, sort_keys=True).encode()).hexdigest()[:20]
SIZES = {"sudoku_hard": (2000, 1024, 976), "text8": (19531, 2048, 2048),
         "tinystories-bpe-10000": (13092, 2048, 2048)}
POLICIES = ["native", "native_greedy", "eb_greedy", "renoising", "renoising_greedy",
            "eb_churn", "psi"]


@pytest.mark.parametrize("task", sorted(SIZES))
def test_partitions_are_disjoint_halving_ready_and_unseen(task):
    spec = json.loads((EVALUATION / f"{task}.json").read_text())
    old = json.loads((ROOT / f"experiments/sc-vs-depth/evaluation/{task}.json").read_text())
    size, pool, test_size = SIZES[task]
    tuning, test = spec["tuning_indices"], spec["test_indices"]
    assert (len(tuning), len(test)) == (pool, test_size)
    assert len(set(tuning)) == pool and not set(tuning) & set(test)
    assert 0 <= min(tuning + test) and max(tuning + test) < size
    # The source runs' training callbacks generated from their tuning indices.
    assert not set(test) & set(old["tuning_indices"])
    assert spec["halving"]["rungs"][-1] <= pool
    assert set(spec["halving"]["finalist_seeds"]).isdisjoint(spec["seeds"] + spec["test_seeds"])
    if task.startswith("tinystories"):
        assert spec["score"] == {"kind": "metric", "metric": "sol"}
        assert spec["direction"] == "minimize"
        sol = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/task/sol/gpt2_large.yaml"))
        assert spec["overrides"]["task.sol"] == sol
        assert spec["overrides"]["task.calculate_bleu"] is True  # BLEU stays recorded
    else:  # Sudoku and text8 keep their metrics and overrides.
        assert (spec["score"], spec["direction"]) == (old["score"], old["direction"])
        assert spec["overrides"] == old.get("overrides", {})


def prepare(sweep, plan, env):
    run_python("-m", "core.sweeps.prepare", "-m", f"+sweep={sweep}",
               f"preparation.output={plan}", env=env)
    run_python("-m", "core.sweeps.prepare", "collect", str(plan), env=env)
    return json.loads((plan / "manifest.json").read_text())


@pytest.mark.parametrize("source,count", [
    ("sc_vs_depth", 27), ("continuous_schedules", 24), ("hybrid_sc", 48),
])
def test_prepare_reuses_source_checkpoints_and_recipes(tmp_path, source, count):
    env = {**os.environ, "PYTHONPATH": str(ROOT), "SDDC_STORAGE_ROOT": str(tmp_path),
           "SDDC_SOURCE_RUNS": str(tmp_path / "h100")}
    cases = prepare(f"{source}_rerun", tmp_path / "plan", env)
    originals = {c["name"]: c for c in prepare(source, tmp_path / "source", env)}
    assert len(cases) == count
    assert {c["name"] for c in cases} == {f"{name}-rerun" for name in originals}
    for case in cases:
        base = originals[case["name"][: -len("-rerun")]]
        assert base["training"] == {"space": {}, "trials": 1}  # one candidate: hash of {}
        assert case["training"] == {
            "space": {}, "trials": 1,
            "checkpoint": str(tmp_path / "h100" / source / base["name"] / "training/candidates"
                              / CANDIDATE / "training/checkpoints/last.ckpt"),
        }
        assert [p["name"] for p in case["inference"]] == POLICIES
        assert case["inference"][:5] == base["inference"]  # the source policies, exactly
        evaluation = case["evaluation"]
        assert evaluation["nfe"] == 128 and evaluation["extra_test_nfe"] == [512]
        assert (evaluation["extra_test_examples"], evaluation["extra_test_seeds"]) == (128, [1041])
        task = next(t for t in SIZES if base["name"].startswith(t))
        assert evaluation["test_indices"] == json.loads(
            (EVALUATION / f"{task}.json").read_text())["test_indices"]
        new, old = OmegaConf.load(case["config"]), OmegaConf.load(base["config"])
        new.callback.output_dir = old.callback.output_dir = None  # under each plan's runs/
        for key in set(new) | set(old):
            if key not in {"output_dir", "wandb", "benchmark"}:
                assert new.get(key) == old.get(key), key  # the checkpoint's recipe, unchanged
    rows, errors = nested_tuning.check_manifest(cases)
    assert not rows and all("fixed checkpoint" in e for e in errors)  # none on this machine


def candidate_settings(policy, trials=4):
    """The policy's initial points and a few random draws from its space."""
    points = list(policy.get("initial_points", []))
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=3))
    points += [nested_tuning.suggest(study.ask(), policy["space"]) for _ in range(trials)]
    return points


@pytest.mark.parametrize("name", ["eb_churn", "psi"])
@pytest.mark.parametrize("family,view", [
    ("gaussian", None), ("simplex", None), ("spherical", None), ("vmf", None),
    ("masked", None), ("uniform", None), ("gaussian", "masked"), ("spherical", "uniform"),
])
def test_new_policies_sample_every_family(name, family, view):
    from core.sweeps.budget import sampler_steps

    policy = OmegaConf.to_container(
        OmegaConf.load(ROOT / f"configs/inference/inference_policies/{name}.yaml")
    )["preparation"]["extra_inference"][name]
    algo = build_algorithm(family, view, "none")
    tokens = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    prompt = torch.tensor([[True, False, False, False, False]] * 2)
    for parameters in candidate_settings(policy):
        config = OmegaConf.create({"sampler": policy["sampler"], "algorithm": {"predictor": {}}})
        for key, value in parameters.items():
            OmegaConf.update(config, key, value, force_add=True)
        for nfe in (128, 512):
            assert sampler_steps(config, instantiate(config.sampler), nfe) == nfe - 1
        config.sampler.steps = 6  # a short run of the same sampler on CPU
        algo.predictor.temperature = parameters.get("algorithm.predictor.temperature", 1.0)
        algo.predictor.top_p = parameters.get("algorithm.predictor.top_p", 1.0)
        torch.manual_seed(0)
        output, info = algo.generate(instantiate(config.sampler), tokens, prompt)
        assert torch.equal(output[prompt], tokens[prompt])
        assert max(info["nfe_per_sample"]) <= 7

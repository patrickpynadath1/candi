"""experiments/sol-rerun: SOL-scored, successive-halving reruns on fixed checkpoints.

Prepares continuous_schedules_sol and hybrid_sc_sol, checks the cases reuse the source
sweeps' recipes and checkpoints unchanged, and runs every new policy's sampler (initial
points and random draws from its space) on a tiny recipe at the rerun's NFE budgets.
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
from core.tests.test_sampler_equivalence import build_algorithm

EVALUATION = ROOT / "experiments/sol-rerun/evaluation/tinystories-bpe-10000.json"
SOURCE_CANDIDATE = "training/candidates/44136fa355b3678a1146/training/checkpoints/last.ckpt"
PATHS = ("gaussian_embedding", "simplex", "spherical", "spherical_vmf")
ARMS = ("none", "expected_embedding")


def prepare(sweep, plan, storage):
    env = {**os.environ, "PYTHONPATH": str(ROOT), "SDDC_STORAGE_ROOT": str(storage)}
    run_python("-m", "core.sweeps.prepare", "-m", f"+sweep={sweep}",
               f"preparation.output={plan}", env=env)
    run_python("-m", "core.sweeps.prepare", "collect", str(plan), env=env)
    return json.loads((plan / "manifest.json").read_text())


def test_evaluation_partitions_are_disjoint_and_sol_scored():
    spec = json.loads(EVALUATION.read_text())
    tuning, test = spec["tuning_indices"], spec["test_indices"]
    assert (len(tuning), len(test)) == (2048, 4096)
    assert not set(tuning) & set(test) and len(set(tuning)) == 2048
    assert 0 <= min(tuning + test) and max(tuning + test) < 13092
    assert spec["score"] == {"kind": "metric", "metric": "sol"} and spec["direction"] == "minimize"
    assert spec["halving"]["rungs"] == [64, 256, 1024]
    sol = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/task/sol/gpt2_large.yaml"))
    assert spec["overrides"]["task.sol"] == sol  # one source of truth for the settings
    assert spec["overrides"]["task.primary_metric"] == "sol"
    assert spec["overrides"]["task.calculate_bleu"] is True  # BLEU stays recorded


@pytest.mark.parametrize("sweep,source,views", [
    ("continuous_schedules_sol", "continuous_schedules", [None]),
    ("hybrid_sc_sol", "hybrid_sc", ["masked", "uniform"]),
])
def test_prepare_reuses_source_checkpoints_and_recipes(tmp_path, sweep, source, views):
    storage = tmp_path / "storage"
    cases = prepare(sweep, tmp_path / "plan", storage)
    originals = {
        c["name"]: c
        for c in prepare(source, tmp_path / "source", storage)
        if c["name"].startswith("tinystories")
    }
    expected = {
        "-".join(p for p in ("tinystories-bpe-10000", view, path, arm, "seed-1") if p) + "-sol"
        for view in views for path in PATHS for arm in ARMS
    }
    assert {c["name"] for c in cases} == expected
    policies = ["native", "native_greedy", "eb_greedy", "renoising", "renoising_greedy", "eb_churn"]
    if views != [None]:
        policies.append("psi")
    for case in cases:
        assert [p["name"] for p in case["inference"]] == policies
        source_name = case["name"][: -len("-sol")]
        root = f"runs/{source.replace('_', '-')}"
        assert case["training"] == {
            "space": {}, "trials": 1,
            "checkpoint": str(storage / root / source_name / SOURCE_CANDIDATE),
        }
        assert case["evaluation"]["nfe"] == [128, 512]
        assert len(case["evaluation"]["tuning_indices"]) == 2048
        new, old = OmegaConf.load(case["config"]), OmegaConf.load(originals[source_name]["config"])
        new.callback.output_dir = old.callback.output_dir = None  # under each plan's runs/
        for key in set(new) | set(old):
            if key not in {"output_dir", "wandb", "benchmark"}:
                assert new.get(key) == old.get(key), key  # the checkpoint's recipe, unchanged
        # The source sweep's five policies are kept exactly.
        assert case["inference"][:5] == originals[source_name]["inference"]
    rows, errors = nested_tuning.check_manifest(cases)
    assert not rows and all("fixed checkpoint" in e for e in errors)  # none on this machine


def candidate_settings(policy, trials=4):
    """The policy's initial points and a few random draws from its space."""
    points = list(policy.get("initial_points", []))
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=3))
    points += [nested_tuning.suggest(study.ask(), policy["space"]) for _ in range(trials)]
    return points


@pytest.mark.parametrize("name,view", [("eb_churn", None), ("eb_churn", "masked"),
                                        ("psi", "masked"), ("psi", "uniform")])
def test_new_policies_sample_at_both_budgets(name, view):
    from core.sweeps.budget import sampler_steps

    policy = OmegaConf.to_container(
        OmegaConf.load(ROOT / f"configs/inference/inference_policies/{name}.yaml")
    )["preparation"]["extra_inference"][name]
    algo = build_algorithm("gaussian", view, "none")
    tokens = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    prompt = torch.tensor([[True, False, False, False, False]] * 2)
    for parameters in candidate_settings(policy):
        config = OmegaConf.create({"sampler": policy["sampler"], "algorithm": {"predictor": {}}})
        for key, value in parameters.items():
            OmegaConf.update(config, key, value, force_add=True)
        for nfe in (128, 512):
            steps = sampler_steps(config, instantiate(config.sampler), nfe)
            assert steps == nfe - 1  # one call per step plus the terminal decode
            config.sampler.steps = 6  # a short run of the same sampler on CPU
            predictor = algo.predictor
            predictor.temperature = parameters.get("algorithm.predictor.temperature", 1.0)
            predictor.top_p = parameters.get("algorithm.predictor.top_p", 1.0)
            torch.manual_seed(0)
            output, info = algo.generate(instantiate(config.sampler), tokens, prompt)
            assert torch.equal(output[prompt], tokens[prompt])
            assert max(info["nfe_per_sample"]) <= config.sampler.steps + 1

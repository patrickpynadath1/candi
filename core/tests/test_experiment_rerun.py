"""experiments/rerun: inference-only reruns of sc-vs-depth, continuous-schedules, hybrid-sc.

Checks the evaluation partitions, prepares the three rerun sweeps next to their source
sweeps (the cases must reuse the source recipes and checkpoints unchanged), and runs
every new policy's sampler on a tiny recipe of each corruption family.
"""

import copy
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
GP = {"algorithm": "gp"}
TPE = {"algorithm": "tpe", "multivariate": True, "group": True}
POLICIES = ["native", "native_greedy", "eb_greedy", "renoising", "renoising_greedy",
            "eb_churn", "psi"]


@pytest.mark.parametrize("task", sorted(SIZES))
def test_partitions_are_disjoint_two_stage_ready_and_unseen(task):
    spec = json.loads((EVALUATION / f"{task}.json").read_text())
    old = json.loads((ROOT / f"experiments/sc-vs-depth/evaluation/{task}.json").read_text())
    size, pool, test_size = SIZES[task]
    tuning, test = spec["tuning_indices"], spec["test_indices"]
    assert (len(tuning), len(test)) == (pool, test_size)
    assert len(set(tuning)) == pool and not set(tuning) & set(test)
    assert 0 <= min(tuning + test) and max(tuning + test) < size
    # The source runs' training callbacks generated from their tuning indices.
    assert not set(test) & set(old["tuning_indices"])
    stages = spec["two_stage"]
    assert (stages["stage1"], stages["stage2"], stages["finalists"]) == (64, 256, 3)
    assert stages["stage1"] + stages["stage2"] <= pool
    assert set(stages["stage2_seeds"]).isdisjoint(spec["seeds"] + spec["test_seeds"])
    if task.startswith("tinystories"):  # ref_w1 tunes (with an interval), SOL tests
        assert stages["stage2_overrides"] == {"task.ref_w1.bootstrap": 200}
        assert spec["direction"] == "minimize"
        assert spec["overrides"]["task.calculate_bleu"] is True  # BLEU stays recorded
    else:  # Sudoku and text8 keep their metrics and overrides.
        score = {k: v for k, v in spec["score"].items() if k != "per_example"}
        assert (score, spec["direction"]) == (old["score"], old["direction"])
        assert spec["overrides"] == old.get("overrides", {})


def test_per_example_paths_average_to_the_task_metrics(tmp_path):
    from dataclasses import asdict

    from core.tasks.text8 import Text8Task  # noqa: F401  (instantiated through config)
    from core.tests.test_text8 import settings_for

    settings = settings_for(tmp_path)
    settings.task.min_word_lengths = [5, 6]
    task = instantiate(settings.task)
    task.prepare_data()
    dataset = task.evaluation_dataset("validation")
    examples = [dataset[i] for i in range(len(dataset))]
    metric = task.create_metric_accumulator("validation")
    noise = torch.randint(8, 35, (len(examples), len(examples[0]["input_ids"])))
    metric.update(noise, examples, list(range(len(examples))))
    result = json.loads(json.dumps(asdict(metric.compute())))
    spec = json.loads((EVALUATION / "text8.json").read_text())
    aggregate = nested_tuning.metric_value(result, spec["score"]["metric"])
    values = nested_tuning.per_example_scores(result, spec["score"], aggregate)
    assert len(values) == len(examples)
    sudoku = json.loads((EVALUATION / "sudoku_hard.json").read_text())
    records = [{"index": i, "exact": i % 2 == 0} for i in range(6)]
    assert nested_tuning.per_example_scores({"records": records}, sudoku["score"], 0.5)
    for task_spec in (spec, sudoku):
        assert task_spec["two_stage"]["stage2_bootstrap"] == 200


def test_tinystories_tunes_by_ref_w1_and_tests_by_sol():
    spec = json.loads((EVALUATION / "tinystories-bpe-10000.json").read_text())
    assert spec["score"] == {"kind": "metric", "metric": "ref_w1"}
    assert spec["test_score"] == {"kind": "metric", "metric": "sol"}
    ref = spec["overrides"]["task.ref_w1"]
    fit = set(ref["fit_indices"])
    old = json.loads((ROOT / "experiments/sc-vs-depth/evaluation/tinystories-bpe-10000.json").read_text())
    assert len(fit) == 128 and not fit & (set(spec["tuning_indices"]) | set(spec["test_indices"])
                                          | set(old["tuning_indices"]))
    assert ref["dictionary_sha256"] == spec["fit_split"]["dictionary_sha256"]
    assert (ref["dictionary_bytes"], ref["level"], ref["window_log"]) == (65536, 10, 23)
    assert spec["two_stage"]["stage2_overrides"] == {"task.ref_w1.bootstrap": 200}
    assert "task.sol" not in spec["overrides"] and "task.sol" in spec["test_overrides"]
    sol = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/task/sol/gpt2_large.yaml"))
    assert spec["test_overrides"]["task.sol"] == sol


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
        # The source policies, exactly, apart from the Optuna sampler the rerun picks.
        searches = {p["name"]: p.pop("search", None) for p in copy.deepcopy(case["inference"])}
        changed = ("search", "initial_points")
        assert [{k: v for k, v in p.items() if k not in changed} for p in case["inference"][:5]] \
            == [{k: v for k, v in p.items() if k not in changed} for p in base["inference"]]
        both = [{"sampler.final_denoise": "greedy"}, {"sampler.final_denoise": "sample"}]
        for policy in case["inference"]:
            if policy["name"] in ("native_greedy", "renoising_greedy"):
                assert policy["initial_points"] == both and policy["trials"] == 2
        assert searches == {
            "native": GP, "native_greedy": None, "eb_greedy": GP, "renoising": GP,
            "renoising_greedy": None, "eb_churn": GP, "psi": TPE,
        }
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

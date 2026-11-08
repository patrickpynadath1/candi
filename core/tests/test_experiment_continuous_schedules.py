"""Worked example experiments/continuous-schedules: prepare, tune one tiny case, analyze.

Prepares the full 24-case grid, shrinks one Gaussian case to a tiny CPU model
on the TinyTask, runs it through nested tuning on a private local Ray cluster
(all five sampler policies, one trial each), and runs analyze.py on the result,
including the schedule, loss-bin and corruption tables from the checkpoint.
"""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "experiments/continuous-schedules"
CASE = "text8-gaussian_embedding-none-seed-1"


def run(*args, cwd=ROOT):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return result.stdout


def tiny_case(plan, tmp_path):
    spec = json.loads((plan / "cases" / CASE / "case.json").read_text())
    config = OmegaConf.load(spec["config"])
    config.task = {"_target_": "core.tests.test_entrypoint.TinyTask", "vocab_size": 5, "length": 8}
    config.model.update(hidden_size=16, n_heads=2, n_blocks=1, cond_dim=16, dropout=0.0)
    config.trainer.update(max_steps=3, accumulate_grad_batches=1, precision="32", limit_val_batches=0)
    config.loader.update(global_batch_size=4, eval_batch_size=2, num_workers=0)
    config.training.warmup_steps = 1
    config.lr_scheduler.warmup_steps = 1
    config.training.performance.compile.enabled = False
    config.checkpoint.every_n_steps = 3
    config.callback.every_n_steps = 1000
    config.time_sampler.warmup = 1  # let the CDCD clock become ready and warp time
    config.wandb = None
    path = tmp_path / "tiny.yaml"
    OmegaConf.save(config, path)
    policies = []
    for policy in spec["inference"]:
        policy["trials"] = 1
        if policy.get("initial_points"):
            policy["initial_points"] = policy["initial_points"][:1]
        policies.append(policy)
    spec.pop("wandb")
    return {
        **spec,
        "config": str(path),
        "evaluation": {
            "tuning_indices": [0, 1],
            "test_indices": [2, 3],
            "seeds": [41],
            "test_seeds": [1041],
            "nfe": 4,
            "direction": "maximize",
            "score": {"kind": "metric", "metric": "accuracy"},
        },
        "inference": policies,
        "resources": {"cpu": 1, "gpu": 0},
    }


def test_prepare_tune_and_analyze_one_tiny_case(tmp_path):
    import ray

    from core.sweeps import nested_tuning

    plan = tmp_path / "plan"
    run("-m", "core.sweeps.prepare", "-m", "+sweep=continuous_schedules", f"preparation.output={plan}")
    run("-m", "core.sweeps.prepare", "collect", str(plan))
    manifest = json.loads((plan / "manifest.json").read_text())
    names = {case["name"] for case in manifest}
    assert len(names) == 24
    for task in ("sudoku_hard", "text8", "tinystories-bpe-10000"):
        for path in ("gaussian_embedding", "simplex", "spherical", "spherical_vmf"):
            for arm in ("none", "expected_embedding"):
                assert f"{task}-{path}-{arm}-seed-1" in names
    case = next(case for case in manifest if case["name"] == CASE)
    # The MI210 policy names; analyze.py reports them as renoise_sample/renoise_greedy.
    assert [p["name"] for p in case["inference"]] == [
        "native", "native_greedy", "eb_greedy", "renoising", "renoising_greedy"]
    assert case["evaluation"]["nfe"] == 128 and len(case["evaluation"]["test_indices"]) == 256
    config = OmegaConf.load(case["config"])
    assert config.trainer.max_steps == 20000
    assert (config.model.n_blocks, config.model.hidden_size) == (8, 512)
    assert config.time_sampler._target_.endswith("CDCDTimeSampler") and config.time_sampler.adaptive

    spec = tiny_case(plan, tmp_path)
    runs = tmp_path / "runs"
    ray.init(address="local", num_cpus=2, include_dashboard=False,
             runtime_env={"env_vars": {"PYTHONPATH": str(ROOT)}})
    try:
        nested_tuning.run_case(spec, runs, concurrency=1)
    finally:
        ray.shutdown()
    assert (runs / CASE / "final/result.json").is_file()

    output = tmp_path / "analysis"
    run(str(EXAMPLE / "analyze.py"), str(runs), "--output", str(output), "--device", "cpu",
        "--bins", "4", "--examples", "2", "--grid", "11", "--samples", "64", "--points", "11")
    rows = json.loads((output / "cdcd_performance.json").read_text())["rows"]
    assert len(rows) == 1
    row = rows[0]
    assert (row["task"], row["method"], row["arm"], row["case"]) == ("text8", "gaussian", "none", CASE)
    assert row["renoise_tried"] and len(row["samplers"]) == 5 and len(row["seeds"]) == 1
    assert row["policy"] in {s["policy"] for s in row["samplers"]}
    schedules = json.loads((output / "cdcd_schedules.json").read_text())
    assert not schedules["errors"] and len(schedules["cases"]) == 1
    key = "text8|gaussian|none"
    assert set(schedules["schedules"][key]) == {"u", "t", "fit", "sigma"}
    assert len(schedules["schedules"][key]["u"]) == 11
    assert schedules["cases"][0]["ready"]
    for split in ("bins", "bins_train"):
        assert [b["u"] for b in schedules[split][key]] == [0.125, 0.375, 0.625, 0.875]
        assert set(schedules[split][key][0]) == {"u", "loss", "se", "fit", "std", "std_nats", "loss_nats"}
    corruption = schedules["corruption"][key]
    assert corruption["vocab"] == 5 and len(corruption["u"]) == len(corruption["rank_formula"]) == 17
    assert corruption["error"][0] == 0.0


def test_expected_table_has_every_published_cell():
    with open(EXAMPLE / "expected.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scores = [r for r in rows if r["quantity"] == "score"]
    assert len(scores) == 24
    assert len({(r["task"], r["method"], r["arm"]) for r in scores}) == 24
    assert all("BLEU without brevity penalty" in r["note"] for r in scores if r["task"] == "tinystories")
    linearity = [r for r in rows if r["quantity"].startswith("linearity_r2_")]
    assert min(float(r["value"]) for r in linearity
               if r["task"] != "sudoku_hard" and r["quantity"] != "linearity_r2_min") >= 0.998

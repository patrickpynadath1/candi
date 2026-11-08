"""experiments/sc-vs-depth on CPU: prepare the sweep, nested-tune one tiny case, analyze.

The tiny case keeps the sweep's recipe, evaluation file and five sampler policies;
only sizes, steps, trials and indices shrink. Ray runs as a fresh local cluster.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from core.sweeps.configuration import ROOT

POLICIES = ["native", "native_greedy", "eb_greedy", "renoise_sample", "renoise_greedy"]
# The prepared (MI210) policy names; analyze.py reports them under POLICIES.
PREPARED = ["native", "native_greedy", "eb_greedy", "renoising", "renoising_greedy"]


def prepare(output, *overrides):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.sweeps.prepare",
            "--multirun",
            "+sweep=sc_vs_depth",
            f"preparation.output={output}",
            *overrides,
        ],
        cwd=ROOT,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    from core.sweeps.prepare import collect

    collect(output)
    return json.loads((Path(output) / "manifest.json").read_text())


def tiny_overrides(tmp_path):
    trials = [f"preparation.inference.{i}.trials=1" for i in range(5)]
    return [
        "task=battery/sudoku_hard",
        "ablation_path=masked_elbo",
        "modifiers/feedback=expected_embedding",
        f"data.cache_dir={tmp_path / 'data'}",
        "data.num_train=8",
        "data.num_valid=8",
        "model.hidden_size=32",
        "model.cond_dim=16",
        "model.n_blocks=1",
        "model.n_heads=4",
        "model.dropout=0.0",
        "loader.global_batch_size=4",
        "loader.eval_batch_size=2",
        "loader.num_workers=0",
        "loader.pin_memory=false",
        "training.warmup_steps=1",
        "training.performance.compile.enabled=false",
        "trainer.precision=32-true",
        "trainer.accumulate_grad_batches=1",
        "trainer.max_steps=2",
        "trainer.val_check_interval=2",
        "trainer.log_every_n_steps=1",
        "checkpoint.every_n_steps=2",
        "callback.every_n_steps=0",
        "preparation.local.resources_per_job.cpu=1",
        "preparation.local.resources_per_job.gpu=0",
        "preparation.evaluation.nfe=4",
        "preparation.evaluation.seeds=[41]",
        "preparation.evaluation.test_seeds=[1041]",
        "+preparation.evaluation.tuning_indices=[0,1]",
        "+preparation.evaluation.test_indices=[2,3]",
        *trials,
    ]


def test_sweep_expands_to_the_published_grid(tmp_path):
    cases = prepare(tmp_path / "plan")
    expected = {
        f"{task}-{path}-{arm}-seed-1"
        for task in ("sudoku_hard", "text8", "tinystories-bpe-10000")
        for path in ("gaussian_embedding", "masked", "uniform_posterior_ce")
        for arm in ("none", "expected_embedding", "double_depth")
    }
    assert {case["name"] for case in cases} == expected
    for case in cases:
        assert [policy["name"] for policy in case["inference"]] == PREPARED
        assert case["evaluation"]["nfe"] == 128
        assert len(case["evaluation"]["test_indices"]) == 256


def test_tiny_case_runs_through_nested_tuning_and_both_analyses(tmp_path):
    ray = pytest.importorskip("ray")
    pytest.importorskip("optuna")
    from core.sweeps import nested_tuning


    (case,) = prepare(tmp_path / "plan", *tiny_overrides(tmp_path))
    assert case["name"] == "sudoku_hard-masked-expected_embedding-seed-1"
    assert [policy["name"] for policy in case["inference"]] == PREPARED
    assert case["wandb"] is None  # No W&B entity configured, so no tracking.
    ray.init(
        address="local",
        num_cpus=2,
        num_gpus=0,
        include_dashboard=False,
        runtime_env={"env_vars": {"PYTHONPATH": str(ROOT)}},
    )
    try:
        nested_tuning.run_case(case, tmp_path / "runs", concurrency=1)
    finally:
        ray.shutdown()
    assert (tmp_path / "runs" / case["name"] / "final" / "result.json").is_file()

    for example, produced in (("sc-vs-depth", "sc_depth_results.json"), ("sc-sampler-choice", "sampler_choice.json")):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments" / example / "analyze.py"),
                str(tmp_path / "runs"),
                "--output",
                str(tmp_path / example),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        (row,) = json.loads((tmp_path / example / produced).read_text())["rows"]
        assert (row["task"], row["method"], row["arm"]) == ("sudoku_hard", "masked", "sc")
    (row,) = json.loads((tmp_path / "sc-vs-depth" / "sc_depth_results.json").read_text())["rows"]
    assert [sampler["policy"] for sampler in row["samplers"]] == POLICIES
    assert row["policy"] in POLICIES and 0 <= row["score"] <= 1
    assert len(row["seeds"]) == 1

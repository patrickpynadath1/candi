"""Tokenizer-ladder example end to end on CPU: prepare, nested tuning, analysis.

One tiny case (vocabulary 260, masked ELBO, no self-conditioning) built from a few
synthetic stories; a fresh local Ray cluster, never an existing one.
"""

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "experiments" / "tokenizer-ladder"
NAMES = ["Amy", "Bob", "Cal", "Dan", "Eve", "Flo", "Gil", "Hal", "Ivy", "Jon", "Kim", "Lou"]
THINGS = ["red ball", "small boat", "blue kite"]


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_stories(directory):
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory.mkdir()
    texts = [
        f"Once upon a time {name} found a {thing} in the park. {name} was happy "
        f"and shared the {thing} with a friend."
        for thing in THINGS
        for name in NAMES
    ]
    pq.write_table(pa.table({"text": texts[:24]}), directory / "train-00000.parquet")
    # One duplicate of a training story: the cohort must drop it from validation.
    pq.write_table(pa.table({"text": texts[23:]}), directory / "validation-00000.parquet")


def run(*command, timeout=120):
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": str(ROOT), "WANDB_MODE": "disabled"}
    for name in ("WANDB_ENTITY", "RAY_ADDRESS"):
        environment.pop(name, None)
    result = subprocess.run(
        [sys.executable, *command], cwd=ROOT, env=environment, capture_output=True, text=True, timeout=timeout
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-6000:]
    return result.stdout


def test_one_tiny_case_prepares_tunes_and_analyzes(tmp_path):
    pytest.importorskip("sacrebleu")
    pytest.importorskip("pyarrow")
    pytest.importorskip("ray")
    import pyarrow.parquet as pq

    write_stories(tmp_path / "parquet")
    # The shared cohort's rule for stories that fit every capacity: all of them,
    # minus validation stories that also occur in training.
    keys = {
        split: {hashlib.sha256(text.encode()).hexdigest() for text in pq.read_table(path).column(0).to_pylist()}
        for split, path in ((s, tmp_path / "parquet" / f"{s}-00000.parquet") for s in ("train", "validation"))
    }
    ids = [f"train:{k}" for k in sorted(keys["train"])]
    ids += [f"validation:{k}" for k in sorted(keys["validation"] - keys["train"])]
    assert len(ids) == 24 + 12
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "allowed_ids.json").write_text(json.dumps(ids))

    # The committed spec, with indices into the tiny validation split.
    spec = json.loads((EXAMPLE / "evaluation" / "tinystories-bpe-260.json").read_text())
    spec.update(tuning_indices=[0, 1], test_indices=[2, 3])
    (tmp_path / "evaluation.json").write_text(json.dumps(spec))
    plan = tmp_path / "plan"
    run(
        "-m", "core.sweeps.prepare", "-m", "+sweep=tokenizer_ladder",
        "bpe.vocabulary_size=260", "ladder_recipe=masked",
        f"preparation.output={plan}", f"preparation.evaluation_file={tmp_path / 'evaluation.json'}",
        "preparation.evaluation.nfe=4", "preparation.evaluation.seeds=[41]",
        "preparation.evaluation.test_seeds=[1041]",
        *(f"preparation.inference.{i}.trials=1" for i in range(5)),
        "preparation.local.resources_per_job={cpu:1,gpu:0}",
        f"data.parquet_dir={tmp_path / 'parquet'}",
        f"data.cache_dir={tmp_path / 'cache'}",
        "task.length=128", "model.hidden_size=16", "model.cond_dim=16", "model.n_heads=2",
        "model.n_blocks=1", "trainer.max_steps=2", "trainer.accumulate_grad_batches=1",
        "trainer.precision=32", "trainer.val_check_interval=1", "trainer.limit_val_batches=1",
        "loader.global_batch_size=2", "loader.eval_batch_size=2", "loader.num_workers=0",
        "training.performance.compile.enabled=false", "callback.every_n_steps=1000",
        "checkpoint.every_n_steps=2",
    )
    run("-m", "core.sweeps.prepare", "collect", str(plan))
    (case,) = json.loads((plan / "manifest.json").read_text())
    assert case["name"] == "tinystories-bpe-260-masked-none-seed-1"
    assert [p["name"] for p in case["inference"]] == [
        "native", "native_greedy", "eb_greedy", "renoising", "renoising_greedy"
    ]
    assert case["evaluation"]["score"] == {"kind": "metric", "metric": "corpus_bleu"}
    weighting = OmegaConf.load(case["config"]).objective.weighting
    assert weighting._target_.endswith("KeepRateHazardWeighting")  # masked ELBO
    # No W&B logger in the test (the tracking extra is not installed here).
    case["wandb"] = None
    (plan / "manifest.json").write_text(json.dumps([case]))

    runs = tmp_path / "runs"
    run(
        "-m", "core.sweeps.nested_tuning", str(plan / "manifest.json"), "--output", str(runs),
        "--address", "local", "--concurrency", "1", timeout=240,
    )
    data = load("analyze").main([str(runs), "--output", str(tmp_path / "analysis")])
    (row,) = data["rows"]
    assert (row["vocab"], row["method"], row["sc"]) == (260, "masked", False)
    assert set(row["samplers"]) == {"native", "native_greedy", "eb_greedy", "renoise_sample", "renoise_greedy"}
    assert row["chosen"] in row["samplers"] and len(row["test_seeds"]) == 1
    assert {"test_nobp", "test_bp", "test_len_ratio"} <= set(row)
    assert json.loads((tmp_path / "analysis" / "vocab_ladder.json").read_text()) == data


def test_sweep_expands_to_the_published_35_case_grid(tmp_path):
    recipes = {value: name for name, value in load("analyze").RECIPES.items()}
    with open(EXAMPLE / "expected.csv") as handle:
        expected = {
            f"tinystories-bpe-{r['vocab']}-"
            f"{recipes[(r['method'], None if r['sc'] == '' else r['sc'] == 'True')]}-seed-1"
            for r in csv.DictReader(handle)
        }
    assert len(expected) == 35
    plan = tmp_path / "plan"
    environment = {**os.environ, "OMP_NUM_THREADS": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "core.sweeps.prepare", "-m", "+sweep=tokenizer_ladder",
         f"preparation.output={plan}"],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    run("-m", "core.sweeps.prepare", "collect", str(plan))
    cases = {c["name"]: c for c in json.loads((plan / "manifest.json").read_text())}
    assert set(cases) == expected
    for name, case in cases.items():
        capacity = {"260": 1024, "500": 768}.get(name.split("-")[2], 512)
        assert OmegaConf.load(case["config"]).task.length == capacity
        evaluation = case["evaluation"]
        assert (len(evaluation["tuning_indices"]), len(evaluation["test_indices"])) == (128, 256)
        assert evaluation["nfe"] == (capacity if "-ar-8-" in name else 128)

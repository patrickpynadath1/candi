"""experiments/block-encoder-decoder: the sweep prepares, one tiny case runs, analysis reads it."""

import importlib.util
import json
import os
import subprocess
import sys

from omegaconf import OmegaConf
import pytest

from core.sweeps.configuration import ROOT
from core.sweeps.prepare import collect

EXAMPLE = ROOT / "experiments/block-encoder-decoder"
SHAPES = {
    "e2d2-encoder-1-decoder-8": ("e2d2", 1, 8), "e2d2-encoder-2-decoder-8": ("e2d2", 2, 8),
    "e2d2-encoder-4-decoder-8": ("e2d2", 4, 8), "e2d2-encoder-8-decoder-8": ("e2d2", 8, 8),
    "e2d2-encoder-8-decoder-1": ("e2d2", 8, 1), "e2d2-encoder-8-decoder-2": ("e2d2", 8, 2),
    "e2d2-encoder-8-decoder-4": ("e2d2", 8, 4), "raw-context-decoder-8": ("raw", 0, 8),
    "shared-transformer-8": ("shared", None, 8),
}
TASKS = ["sudoku_hard", "text8", "tinystories-bpe-10000"]


def load(name, directory=None):
    spec = importlib.util.spec_from_file_location(name, (directory or EXAMPLE) / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment():
    env = {**os.environ, "OMP_NUM_THREADS": "1", "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0"}
    for key in ("RAY_ADDRESS", "RAY_JOB_CONFIG_JSON_ENV_VAR", "WANDB_ENTITY"):
        env.pop(key, None)
    return env


def prepare(output, *overrides):
    result = subprocess.run(
        [sys.executable, "-m", "core.sweeps.prepare", "-m", "+sweep=block_encoder_decoder",
         f"preparation.output={output}", *overrides],
        cwd=ROOT, env=environment(), capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return collect(output)


def test_sweep_prepares_the_27_published_cases(tmp_path):
    assert prepare(tmp_path / "plan") == 27
    cases = json.loads((tmp_path / "plan/manifest.json").read_text())
    expected = {f"{task}-{recipe}-seed-1" for task in TASKS for recipe in SHAPES}
    assert {case["name"] for case in cases} == expected
    architecture = load("analyze").architecture
    for case in cases:
        task = next(t for t in TASKS if case["name"].startswith(t + "-"))
        recipe = case["name"][len(task) + 1 : -len("-seed-1")]
        spec = json.loads((EXAMPLE / f"evaluation/{task}.json").read_text())
        for key in ("tuning_indices", "test_indices", "score", "seeds", "test_seeds"):
            assert case["evaluation"][key] == spec[key]
        assert [p["name"] for p in case["inference"]] == ["native", "native_greedy"]
        config = OmegaConf.to_container(OmegaConf.load(case["config"]), resolve=True)
        assert architecture(config) == SHAPES[recipe]
        assert config["algorithm"]["time_conditioning"] is True
        assert config["block_diffusion"]["block_size"] == 32
        assert config["trainer"]["max_steps"] == 20000
        assert config["training"]["performance"]["activation_checkpointing"] is True


def test_one_tiny_case_runs_through_nested_tuning_and_analysis(tmp_path):
    # Sudoku-hard with a tiny generated dataset; real recipe, CPU-sized model.
    spec = tmp_path / "sudoku_hard.json"
    spec.write_text(json.dumps({"task": "sudoku_hard", "tuning_indices": [0], "test_indices": [1],
                                "score": {"kind": "metric", "metric": "accuracy"}, "direction": "maximize"}))
    plan = tmp_path / "plan"
    assert prepare(
        plan, "task=battery/sudoku_hard", "block=encoder1_decoder8",
        f"preparation.evaluation_file={spec}", "preparation.evaluation.nfe=12",
        "preparation.local.resources_per_job={cpu:1,gpu:0}",
        f"data.cache_dir={tmp_path / 'data'}", "data.num_train=4", "data.num_valid=2",
        "model.hidden_size=32", "model.cond_dim=16", "model.n_heads=4",
        "trainer.accelerator=cpu", "trainer.precision=32-true", "trainer.max_steps=2",
        "trainer.accumulate_grad_batches=1", "trainer.log_every_n_steps=1",
        "loader.global_batch_size=2", "loader.eval_batch_size=2", "loader.num_workers=0",
        "loader.pin_memory=false", "training.warmup_steps=1", "training.performance.compile.enabled=false",
        "checkpoint.every_n_steps=1", "callback.every_n_steps=0", "evaluation.device=cpu",
    ) == 1
    manifest = plan / "manifest.json"
    cases = json.loads(manifest.read_text())
    for case in cases:
        # One seed and one trial per policy; no W&B tracking.
        case["evaluation"].update(seeds=[41], test_seeds=[1041])
        for policy in case["inference"]:
            policy["trials"] = 1
        del case["wandb"]
        config = OmegaConf.load(case["config"])
        config.wandb = None
        OmegaConf.save(config, case["config"])
    manifest.write_text(json.dumps(cases))
    runs = tmp_path / "runs"
    result = subprocess.run(
        [sys.executable, "-m", "core.sweeps.nested_tuning", str(manifest), "--output", str(runs),
         "--address", "local", "--concurrency", "1"],
        cwd=ROOT, env=environment(), capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    result = subprocess.run(
        [sys.executable, str(EXAMPLE / "analyze.py"), str(runs), "--output", str(tmp_path / "out")],
        cwd=ROOT, env=environment(), capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    (row,) = json.loads((tmp_path / "out/block_scaling.json").read_text())["rows"]
    assert (row["task"], row["kind"], row["enc"], row["dec"]) == ("sudoku_hard", "e2d2", 1, 8)
    assert row["status"] == "completed" and row["policy"] in ("native", "native_greedy")
    assert 0 <= row["score"] <= 1 and len(row["seeds"]) == 1
    assert row["train_hours"] >= 0 and row["eval_seconds"] > 0 and row["nfe"] > 0


def test_data_preparation_rebuilds_the_cohort_rule(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("tokenizers")
    data = load("prepare_data", ROOT / "scripts")
    story = "Once upon a time, there was a little girl named Lily who loved to play outside. "
    shared, long = story + "She smiled.", story * 40
    texts = {"train": [shared, story + "The end.", long, "Too short."],
             "validation": [shared, story + "Bye.", story + "Bye."]}
    for split, rows in texts.items():
        pq.write_table(pa.table({"text": rows}), tmp_path / f"{split}-00000-of-00001.parquet")
    ids = data.cohort(tmp_path)
    key = data.split_story
    # Short and over-long stories are dropped, duplicates collapse, and a
    # validation story also in train stays only in train.
    assert ids == sorted([f"train:{key(shared)[2]}", f"train:{key(story + 'The end.')[2]}",
                          f"validation:{key(story + 'Bye.')[2]}"])
    raw = tmp_path / "raw"
    raw.mkdir()
    for split in data.TEXT8_SHA256:
        (raw / f"text8.{split}.txt").write_text("anarchism originated")
    assert data.check_text8(raw) == ["train", "validation", "test"]

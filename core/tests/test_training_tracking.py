"""Exercise periodic generation logging with the real offline W&B SDK."""

import json

import pytest
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from core.tests.test_mdlm import config
from core.training.runner import run


@pytest.mark.parametrize("multiple", [False, True])
def test_training_logs_periodic_generations_and_finishes_wandb(
    tmp_path, monkeypatch, multiple
):
    wandb = pytest.importorskip("wandb")
    settings = config()
    settings.task = OmegaConf.create(
        {
            "_target_": "core.tests.test_entrypoint.TinyTask",
            "vocab_size": 5,
            "length": 8,
        }
    )
    settings.output_dir = str(tmp_path)
    settings.callback.every_n_steps = 2
    if multiple:
        settings.callback.sampler_configs = {
            "first": settings.sampler,
            "second": settings.sampler,
        }
    namespace = "tiny/first" if multiple else "tiny"
    settings.trainer.enable_progress_bar = False
    settings.wandb = {"project": "candi-test", "mode": "offline"}
    logged_metrics, tables, artifacts, runs, direct_logs = [], [], [], [], []
    original_metrics = WandbLogger.log_metrics
    original_table = WandbLogger.log_table
    original_init = wandb.init

    def log_metrics(self, metrics, step=None):
        logged_metrics.append((dict(metrics), step))
        return original_metrics(self, metrics, step=step)

    def log_table(self, key, **kwargs):
        tables.append((key, kwargs))
        return original_table(self, key, **kwargs)

    def init(*args, **kwargs):
        tracked = original_init(*args, **kwargs)
        runs.append(tracked)
        original_log = tracked.log

        def log(data, *args, **kwargs):
            direct_logs.append(dict(data))
            return original_log(data, *args, **kwargs)

        monkeypatch.setattr(tracked, "log", log)
        original_artifact = tracked.log_artifact

        def log_artifact(artifact, *args, **kwargs):
            artifacts.append((artifact.name, dict(artifact.metadata)))
            return original_artifact(artifact, *args, **kwargs)

        monkeypatch.setattr(tracked, "log_artifact", log_artifact)
        return tracked

    monkeypatch.setattr(wandb, "init", init)
    monkeypatch.setattr(WandbLogger, "log_metrics", log_metrics)
    monkeypatch.setattr(WandbLogger, "log_table", log_table)
    run(settings)

    assert wandb.run is None
    assert len(runs) == 1
    assert any("train/loss" in values for values, _ in logged_metrics)
    assert any(
        f"{namespace}/accuracy" in values and step == 2
        for values, step in logged_metrics
    )
    assert tables[0][0] == f"{namespace}/samples"
    assert tables[0][1]["step"] == 2
    subdir = "task_callback/first" if multiple else "task_callback"
    result = json.loads((tmp_path / subdir / "step_00000002.json").read_text())
    assert len(tables[0][1]["data"]) == len(result["records"])
    artifact_name = "tiny-first-generation" if multiple else "tiny-generation"
    assert (artifact_name, {"step": 2, "task": "tiny"}) in artifacts
    csv = next((tmp_path / "metrics").glob("*/metrics.csv")).read_text()
    assert f"{namespace}/accuracy" in csv
    assert list((tmp_path / "wandb").glob("offline-run-*/*.wandb"))
    assert (tmp_path / "checkpoints/last.ckpt").exists()

    if multiple:
        assert any(
            "tiny/second/accuracy" in values and step == 2
            for values, step in logged_metrics
        )
    settings.mode = "evaluate"
    settings.evaluation.checkpoint = str(tmp_path / "checkpoints/last.ckpt")
    settings.evaluation.device = "cpu"
    settings.evaluation.policy = "final_policy"
    settings.output_dir = str(tmp_path / "final")
    run(settings)
    assert wandb.run is None
    assert len(runs) == 2
    assert any("tiny/final_policy/accuracy" in values for values in direct_logs)
    assert (tmp_path / "final/results.json").exists()

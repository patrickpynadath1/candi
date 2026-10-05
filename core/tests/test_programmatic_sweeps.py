"""Hydra/Tune integration contracts; scheduling itself belongs to Ray."""

import json
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from core.sweeps import compose, run_trial
from core.sweeps.tune import _metrics, _prepare


def tiny():
    config = compose("smoke")
    config.task = {
        "_target_": "core.tests.test_entrypoint.TinyTask",
        "vocab_size": 5,
        "length": 8,
    }
    config.model.hidden_size = 16
    config.model.n_heads = 2
    config.trainer.max_steps = 1
    config.trainer.enable_progress_bar = False
    config.trainer.limit_val_batches = 0
    config.checkpoint.every_n_steps = 1
    config.loader.num_workers = 0
    return OmegaConf.to_container(config, resolve=False)


@pytest.mark.parametrize(
    "method", sorted(path.stem for path in Path("configs/experiment").glob("*.yaml"))
)
def test_all_presets_resolve(method, tmp_path):
    cfg = _prepare({"experiment": method}, tmp_path, {"GPU": 1})
    from hydra.utils import get_class

    algorithm = get_class(cfg.algorithm._target_)
    assert callable(algorithm.loss) and callable(algorithm.generate)
    assert "${" not in OmegaConf.to_yaml(cfg)
    assert cfg.output_dir == str(tmp_path)


def test_editable_config_and_trial_isolation(tmp_path):
    cfg = compose("gaussian_diffusion_sudoku", {"sampler": "heun"})
    assert cfg.sampler._target_.endswith("HeunSampler")
    cfg.model.hidden_size = 96
    spec = {"config": OmegaConf.to_container(cfg, resolve=False)}
    first = _prepare(spec, tmp_path / "first", {"GPU": 1})
    second = _prepare(spec, tmp_path / "second", {"CPU": 1})
    assert first.algorithm.model.hidden_size == 96
    assert first.evaluation.device == "cuda:0"
    assert second.evaluation.device == "cpu"
    assert first.output_dir != second.output_dir
    assert cfg.output_dir not in (first.output_dir, second.output_dir)
    first.task.length = 32
    assert second.task.length != 32


def test_mapping_overrides():
    config = compose(
        "sudoku_hard",
        {
            "++model.input_adapter": {"_target_": "core.models.components.TokenInput"},
            "++example": [{"key": "value"}, 2],
        },
    )
    assert config.example[0].key == "value"


@pytest.mark.parametrize("gpus", [0.5, 2])
def test_unsupported_gpu_allocations(gpus, tmp_path):
    with pytest.raises(ValueError, match="whole GPU"):
        _prepare({"config": tiny()}, tmp_path, {"GPU": gpus})


def test_bad_config_rejected(tmp_path):
    with pytest.raises(ValueError, match="either"):
        _prepare({"config": tiny(), "experiment": "smoke"}, tmp_path, {})
    with pytest.raises(ValueError, match="Unknown"):
        _prepare({"seed": 2}, tmp_path, {})
    cfg = tiny()
    cfg["trainer"]["devices"] = 2
    with pytest.raises(ValueError, match="devices=1"):
        _prepare({"config": cfg}, tmp_path, {})


def test_evaluation_metrics_exclude_records(tmp_path):
    (tmp_path / "results.json").write_text(
        json.dumps(
            {
                "accuracy": 0.5,
                "nested": {"entropy": 2.0},
                "records": [{"text": "sample"}],
            }
        )
    )
    assert _metrics(tmp_path) == {"accuracy": 0.5, "nested/entropy": 2.0}


def test_real_ray_training_failure_and_restore(tmp_path, monkeypatch):
    monkeypatch.setenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    import ray
    from ray import tune

    ray.init(
        num_cpus=2,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=100 * 1024**2,
    )
    try:
        bad = tiny()
        bad["mode"] = "invalid"
        trainable = tune.with_resources(run_trial, {"cpu": 1})
        tuner = tune.Tuner(
            trainable,
            param_space={
                "trial": tune.grid_search([{"config": tiny()}, {"config": bad}])
            },
            run_config=tune.RunConfig(
                storage_path=str(tmp_path), name="integration", verbose=0
            ),
        )
        results = tuner.fit()
        assert len(results) == 2
        assert results.num_errors == 1
        success = next(result for result in results if result.error is None)
        output = Path(success.metrics["output_dir"])
        checkpoint = output / "checkpoints/last.ckpt"
        assert checkpoint.exists()
        assert success.metrics["cuda_visible_devices"] == ""
        assert "train/loss" in success.metrics
        with success.checkpoint.as_directory() as saved:
            assert Path(saved, "checkpoints/last.ckpt").exists()
            assert Path(saved, "config.yaml").exists()
        before = checkpoint.stat().st_mtime_ns
        restored = tune.Tuner.restore(
            str(tmp_path / "integration"), trainable=trainable
        ).fit()
        assert len(restored) == 2
        assert checkpoint.stat().st_mtime_ns == before
    finally:
        ray.shutdown()

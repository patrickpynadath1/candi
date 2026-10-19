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


@pytest.mark.parametrize("gpus", [-1, 0.5, float("nan"), float("inf")])
def test_unsupported_gpu_allocations(gpus, tmp_path):
    with pytest.raises(ValueError, match="whole GPU"):
        _prepare({"config": tiny()}, tmp_path, {"GPU": gpus})


def test_bad_config_rejected(tmp_path):
    with pytest.raises(ValueError, match="either"):
        _prepare({"config": tiny(), "experiment": "smoke"}, tmp_path, {})
    with pytest.raises(ValueError, match="Unknown"):
        _prepare({"seed": 2}, tmp_path, {})
    cfg = tiny()
    cfg["trainer"]["num_nodes"] = 2
    with pytest.raises(ValueError, match="one node"):
        _prepare({"config": cfg}, tmp_path, {})


@pytest.mark.parametrize("gpus", [0, 1, 2, 8])
def test_gpu_reservation_controls_devices(gpus, tmp_path):
    source = tiny()
    cfg = _prepare({"config": source}, tmp_path, {"GPU": gpus})
    assert cfg.trainer.devices == max(1, gpus)
    assert (
        cfg.trainer.accumulate_grad_batches
        == source["trainer"]["accumulate_grad_batches"]
    )
    assert cfg.loader.global_batch_size == source["loader"]["global_batch_size"]
    assert cfg.trainer.strategy == ("ddp" if gpus > 1 else "auto")
    assert source["trainer"]["devices"] == 1


def test_multi_gpu_explicit_ddp_and_rejections(tmp_path):
    cfg = compose("tinygsm_blockgen_masked", {"strategy": "ddp"})
    prepared = _prepare({"config": cfg}, tmp_path, {"GPU": 8})
    assert prepared.strategy.find_unused_parameters
    cfg.strategy.start_method = "spawn"
    with pytest.raises(ValueError, match="start_method"):
        _prepare({"config": cfg}, tmp_path, {"GPU": 8})
    cfg.strategy = None
    cfg.mode = "evaluate"
    with pytest.raises(ValueError, match="training only"):
        _prepare({"config": cfg}, tmp_path, {"GPU": 8})


def test_torchrun_preserves_ray_visibility_and_propagates_failure(
    tmp_path, monkeypatch
):
    import subprocess
    from core.sweeps import tune as adapter

    cfg = _prepare({"config": tiny()}, tmp_path, {"GPU": 2})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,6")

    class FailedProcess:
        def __init__(self, command, **kwargs):
            assert "--nproc_per_node=2" in command
            assert "core.sweeps.distributed_worker" in command
            assert "env" not in kwargs  # Inherits the actor's GPU isolation.
            import os

            assert os.environ["CUDA_VISIBLE_DEVICES"] == "3,6"
            saved = OmegaConf.load(command[-1])
            assert saved.trainer.devices == 2

        def wait(self):
            return 7

        def poll(self):
            return 7

    monkeypatch.setattr(adapter.subprocess, "Popen", FailedProcess)
    with pytest.raises(subprocess.CalledProcessError) as error:
        adapter._run_distributed(cfg)
    assert error.value.returncode == 7


def test_real_torchrun_training_and_checkpoint_resume(tmp_path):
    from core.sweeps.tune import _run_distributed
    import torch

    cfg = _prepare({"config": tiny()}, tmp_path, {"GPU": 2})
    # Exercise the actual two-rank launch/checkpoint path without requiring GPUs.
    cfg.trainer.accelerator = "cpu"
    cfg.trainer.precision = "32-true"
    cfg.loader.global_batch_size = 4
    cfg.trainer.accumulate_grad_batches = 1
    _run_distributed(cfg)
    checkpoint = tmp_path / "checkpoints/last.ckpt"
    assert torch.load(checkpoint, weights_only=False)["global_step"] == 1
    cfg.checkpoint.resume = str(checkpoint)
    cfg.trainer.max_steps = 2
    _run_distributed(cfg)
    assert torch.load(checkpoint, weights_only=False)["global_step"] == 2


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
        _node_ip_address="127.0.0.1",
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


def test_retry_uses_own_latest_checkpoint(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ray import tune
    from core.training import runner

    output = tmp_path / "run"
    checkpoint = output / "checkpoints/last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    context = SimpleNamespace(
        get_trial_dir=lambda: str(tmp_path),
        get_storage=lambda: SimpleNamespace(experiment_fs_path=str(tmp_path)),
        get_trial_resources=lambda: SimpleNamespace(required_resources={"CPU": 1}),
    )
    monkeypatch.setattr(tune, "get_context", lambda: context)
    config = tiny()
    config["checkpoint"]["resume"] = "/old/source/last.ckpt"

    def inspect(config):
        assert config.checkpoint.resume == str(checkpoint)
        raise RuntimeError("checkpoint verified")

    monkeypatch.setattr(runner, "run", inspect)
    with pytest.raises(RuntimeError, match="checkpoint verified"):
        run_trial({"config": config})


@pytest.mark.parametrize(
    "local_step,requested_step", [(230000, 231251), (240000, 231251)]
)
def test_resume_chooses_newest_checkpoint(
    tmp_path, monkeypatch, local_step, requested_step
):
    from types import SimpleNamespace
    import torch
    from ray import tune
    from core.training import runner

    local = tmp_path / "run/checkpoints/last.ckpt"
    local.parent.mkdir(parents=True)
    requested = tmp_path / "best.ckpt"
    torch.save({"global_step": local_step}, local)
    torch.save({"global_step": requested_step}, requested)
    context = SimpleNamespace(
        get_trial_dir=lambda: str(tmp_path),
        get_storage=lambda: SimpleNamespace(experiment_fs_path=str(tmp_path)),
        get_trial_resources=lambda: SimpleNamespace(required_resources={"CPU": 1}),
    )
    monkeypatch.setattr(tune, "get_context", lambda: context)
    config = tiny()
    config["checkpoint"]["resume"] = str(requested)

    def inspect(config):
        expected = local if local_step >= requested_step else requested
        assert config.checkpoint.resume == str(expected)
        raise RuntimeError("checkpoint verified")

    monkeypatch.setattr(runner, "run", inspect)
    with pytest.raises(RuntimeError, match="checkpoint verified"):
        run_trial({"config": config})


def test_csv_metrics_preserve_booleans_and_numeric_values(tmp_path):
    directory = tmp_path / "metrics/version_0"
    directory.mkdir(parents=True)
    (directory / "metrics.csv").write_text(
        "step,train/loss,head,sc\n9,1.2,True,False\n10,0.8,,\n"
    )
    assert _metrics(tmp_path) == {
        "step": 10.0,
        "train/loss": 0.8,
        "head": True,
        "sc": False,
    }


def test_pending_cancellation_returns_before_training(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ray import tune
    from core.training import runner
    import json

    config = tiny()
    config["wandb"] = {"id": "cancel-me"}
    (tmp_path / "queue-control.json").write_text(
        json.dumps({"cancelled_wandb_ids": ["cancel-me"]})
    )
    context = SimpleNamespace(
        get_trial_dir=lambda: str(tmp_path),
        get_trial_resources=lambda: SimpleNamespace(required_resources={"CPU": 1}),
        get_storage=lambda: SimpleNamespace(experiment_fs_path=str(tmp_path)),
    )
    monkeypatch.setattr(tune, "get_context", lambda: context)
    reported = []
    monkeypatch.setattr(tune, "report", lambda result: reported.append(result))
    def unexpected(config):
        raise AssertionError("Cancelled trial must not start training")
    monkeypatch.setattr(runner, "run", unexpected)
    run_trial({"config": config})
    assert reported == [{"cancelled": True, "cancelled_run_id": "cancel-me"}]

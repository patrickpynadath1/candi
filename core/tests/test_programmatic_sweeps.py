"""Config composition and local execution contracts for programmable sweeps."""

import json
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from core.sweeps import Resources, Sweep, SweepFailed, Trial, compose, grid, run_sweep
from core.sweeps import local as execution


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
    return config


@pytest.mark.parametrize(
    "method",
    [
        "autoregressive",
        "blurring_diffusion",
        "duo",
        "replaid",
        "spherical_flow_matching",
        "simplex",
    ],
)
def test_method_presets_resolve_without_hydra_runtime(method, tmp_path):
    plan = Sweep([Trial(method)], output_dir=tmp_path).plan()[0]
    assert plan["config"]["algorithm"]["_target_"].startswith("core.algorithms.")
    assert "${" not in json.dumps(plan["config"])


def test_live_config_group_composition_grid_independence(tmp_path):
    cfg = compose("gaussian_diffusion_sudoku", {"sampler": "heun"})
    assert cfg.sampler._target_.endswith("HeunSampler")
    cfg.model.hidden_size = 96
    assert cfg.algorithm.model.hidden_size == 96
    cfg.task.length = 32
    copied = Trial(config=cfg).compose()
    copied.task.length = 64
    assert cfg.task.length == 32
    trials = grid(
        ["autoregressive", "duo"], {"optimizer.lr": [1e-4, 3e-4]}, seeds=[1, 2]
    )
    assert len(trials) == 8
    trials[0].overrides["optimizer.lr"] = 0.1
    assert trials[1].overrides["optimizer.lr"] == 1e-4
    plans = Sweep(trials, output_dir=tmp_path).run(dry_run=True)
    assert len(plans) == 8
    assert not (tmp_path / "manifest.json").exists()
    assert cfg.output_dir != str(tmp_path)


def test_resources_visible_subset(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,3,5")
    assert Resources(cpus=1).slots() == ["GPU-a", "3", "5"]
    assert Resources(cpus=1, gpu_ids=[5, "GPU-a"], max_parallel=1).slots() == ["5"]
    with pytest.raises(ValueError, match="subset"):
        Resources(gpu_ids=[0]).slots()
    with pytest.raises(ValueError, match="duplicate"):
        Resources(gpu_ids=[3, 3]).slots()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="No GPUs"):
        Resources().slots()
    assert Resources(gpus=0, cpus=1, max_parallel=2).slots() == [None, None]
    with pytest.raises(ValueError, match="multi-GPU"):
        Resources(gpus=2).slots()


def test_invalid_trials(tmp_path):
    with pytest.raises(ValueError, match="either"):
        Trial("duo", config=tiny()).compose()
    with pytest.raises(ValueError, match="duplicate"):
        Sweep([Trial(name="same"), Trial(name="same")], output_dir=tmp_path).plan()
    cfg = tiny()
    cfg.trainer.devices = 2
    with pytest.raises(ValueError, match="devices=1"):
        Sweep([cfg], output_dir=tmp_path).plan()


def test_real_cpu_training_resume_and_conflict(tmp_path, monkeypatch):
    sweep = Sweep(
        [Trial(config=tiny(), name="small")],
        output_dir=tmp_path / "space in path",
        resources=Resources(gpus=0, cpus=1, max_parallel=1),
    )
    result = sweep.run()[0]
    assert result.status == "succeeded"
    assert Path(result.output_dir, "checkpoints/last.ckpt").exists()
    assert result.metrics
    monkeypatch.setattr(
        execution.subprocess,
        "Popen",
        lambda *a, **kw: pytest.fail("Completed trial relaunched"),
    )
    assert sweep.run()[0].status == "succeeded"
    with pytest.raises(ValueError, match="already has a manifest"):
        sweep.run(resume=False)
    sweep.trials[0].config.seed += 1
    with pytest.raises(ValueError, match="differ"):
        sweep.run()


def test_failed_trial_retries_and_reports(tmp_path):
    cfg = tiny()
    cfg.mode = "invalid"
    sweep = Sweep(
        [cfg], output_dir=tmp_path, resources=Resources(gpus=0, cpus=1, max_parallel=1)
    )
    for _ in range(2):
        with pytest.raises(SweepFailed) as error:
            sweep.run()
        assert error.value.results[0].status == "failed"
        assert error.value.results[0].returncode != 0
    assert "mode must be" in (tmp_path / "0000-trial/run.log").read_text()


def test_gpu_queue_allocations_and_reuse(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    launched = []
    active_devices = set()

    class FakeProcess:
        def __init__(self, args, **kwargs):
            self.device = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
            assert self.device not in active_devices
            active_devices.add(self.device)
            self.polls = 0
            self.returncode = None
            launched.append(self.device)

        def poll(self):
            self.polls += 1
            if self.polls > 1:
                active_devices.discard(self.device)
                self.returncode = 0
            return self.returncode

    monkeypatch.setattr(execution.subprocess, "Popen", FakeProcess)
    result = run_sweep(
        [Trial(config=tiny()) for _ in range(5)],
        output_dir=tmp_path,
        resources=Resources(cpus=1),
    )
    assert len(result) == 5
    assert launched == ["2", "3", "2", "3", "2"]
    assert not active_devices


def test_lock_and_interrupt_cleanup(tmp_path, monkeypatch):
    import fcntl

    with (tmp_path / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another coordinator"):
            Sweep(
                [tiny()],
                output_dir=tmp_path,
                resources=Resources(gpus=0, cpus=1, max_parallel=1),
            ).run()
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    monkeypatch.setattr(execution.subprocess, "Popen", lambda *a, **kw: process)
    original_sleep = execution.time.sleep

    def interrupt(_):
        monkeypatch.setattr(execution.time, "sleep", original_sleep)
        raise KeyboardInterrupt

    monkeypatch.setattr(execution.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        Sweep(
            [tiny()],
            output_dir=tmp_path,
            resources=Resources(gpus=0, cpus=1, max_parallel=1),
        ).run()
    assert process.poll() == -signal.SIGTERM
    assert (
        json.loads((tmp_path / "manifest.json").read_text())["trials"][0]["status"]
        == "interrupted"
    )


def test_mapping_overrides_and_existing_outputs(tmp_path):
    config = compose(
        "sudoku_hard",
        {
            "++model.input_adapter": {"_target_": "core.models.components.TokenInput"},
            "++example": [{"key": "value"}, 2],
        },
    )
    assert config.model.input_adapter._target_ == "core.models.components.TokenInput"
    assert config.example[0].key == "value"
    (tmp_path / "0000-trial").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        Sweep([tiny()], output_dir=tmp_path, resources=Resources(gpus=0, cpus=1)).run()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gpus": 1.5},
        {"gpus": True},
        {"cpus": 0.5},
        {"max_parallel": False},
        {"gpus": 2},
    ],
)
def test_invalid_resources_fail_without_device_probe(kwargs):
    with pytest.raises(ValueError):
        Resources(**kwargs)


def test_evaluation_metrics_exclude_records_and_flatten(tmp_path):
    (tmp_path / "results.json").write_text(
        json.dumps(
            {
                "accuracy": 0.5,
                "nested": {"entropy": 2.0},
                "records": [{"text": "sample"}],
            }
        )
    )
    assert execution._metrics(tmp_path) == {"accuracy": 0.5, "nested/entropy": 2.0}

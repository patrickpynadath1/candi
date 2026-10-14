"""Evaluation CLIs retain artifact caching while Tune owns GPU scheduling."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from examples import evaluate_adaptive_sweep as adaptive
from examples import hint_removal_sweep as hints


@pytest.fixture
def fake_ray(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    state = SimpleNamespace(calls=[], fail=False)

    def init(**options):
        state.calls.append(("init", options, os.environ["CUDA_VISIBLE_DEVICES"]))

    class Tuner:
        def __init__(self, trainable, **options):
            state.trainable, state.options = trainable, options

        def fit(self):
            if state.fail:
                raise RuntimeError("driver failed")
            return [
                SimpleNamespace(
                    config={"task": task},
                    error=RuntimeError("worker failed")
                    if task["name"] == "failed"
                    else None,
                )
                for task in state.options["param_space"]["task"]
            ]

    tune = SimpleNamespace(
        Tuner=Tuner,
        with_resources=lambda fn, resources: (fn, resources),
        grid_search=lambda tasks: tasks,
        TuneConfig=lambda **options: options,
        RunConfig=lambda **options: options,
    )
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(
            tune=tune,
            init=init,
            shutdown=lambda: state.calls.append(("shutdown",)),
            is_initialized=lambda: False,
        ),
    )
    return state


def prepare_tasks(tmp_path, script, monkeypatch):
    tasks = []
    for name in ("cached", "pending", "failed"):
        dest = tmp_path / name
        dest.mkdir()
        tasks.append({"name": name, "directory": str(dest)})
    if script is adaptive:
        monkeypatch.setattr(script, "OUTPUT", tmp_path)
        (tmp_path / "tasks.json").write_text(json.dumps(tasks))
        (tmp_path / "cached/results.json").write_text("{}")
        monkeypatch.setattr(script, "compile_results", lambda: None)
        return lambda: script.launch([3, 7])
    for fraction in hints.FRACTIONS:
        (tmp_path / "cached" / f"hints_{fraction:g}.json").write_text("{}")
    # A partially cached configuration still runs exactly once for all fractions.
    (tmp_path / "pending/hints_1.json").write_text("{}")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"tasks": tasks, "fractions": hints.FRACTIONS})
    )
    monkeypatch.setattr(script, "compile_results", lambda output: None)
    return lambda: script.launch(argparse.Namespace(output=tmp_path, gpus="3,7"))


@pytest.mark.parametrize("script", [adaptive, hints])
def test_launch_uses_selected_gpus_and_preserves_cache_and_failures(
    script, tmp_path, monkeypatch, fake_ray
):
    launch = prepare_tasks(tmp_path, script, monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="1 .* fail"):
        launch()
    assert fake_ray.calls[0][2] == "3,7"
    assert fake_ray.calls[0][1]["num_gpus"] == 2
    assert fake_ray.calls[0][1]["_temp_dir"] == str(tmp_path / "home/.cache/duo/ray")
    assert (tmp_path / "home/.cache/duo/ray").is_dir()
    assert fake_ray.trainable == (script.evaluate_trial, {"cpu": 2, "gpu": 1})
    assert [task["name"] for task in fake_ray.options["param_space"]["task"]] == [
        "pending",
        "failed",
    ]
    assert fake_ray.options["tune_config"] == {
        "max_concurrent_trials": 2,
        "reuse_actors": False,
    }
    assert [
        task["name"] for task in json.loads((tmp_path / "failures.json").read_text())
    ] == ["failed"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert fake_ray.calls[-1] == ("shutdown",)


@pytest.mark.parametrize("script", [adaptive, hints])
def test_driver_failure_restores_gpu_visibility(
    script, tmp_path, monkeypatch, fake_ray
):
    launch = prepare_tasks(tmp_path, script, monkeypatch)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    fake_ray.fail = True
    with pytest.raises(RuntimeError, match="driver failed"):
        launch()
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert fake_ray.calls[-1] == ("shutdown",)


@pytest.mark.parametrize("script", [adaptive, hints])
def test_trial_preserves_worker_gpu_assignment_and_logs(script, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setattr(
        subprocess, "run", lambda command, **options: calls.append((command, options))
    )
    config = {"task": {"directory": str(tmp_path)}, "output": str(tmp_path)}
    (tmp_path / "evaluation.log").write_text("old log\n")
    script.evaluate_trial(config)
    command, options = calls[0]
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == "7"
    assert options["check"] is True
    assert "--worker" in command
    assert options["cwd"] == script.ROOT
    assert (tmp_path / "evaluation.log").read_text() == (
        "old log\n" if script is hints else ""
    )
    if script is hints:
        assert command[command.index("--output") + 1] == str(tmp_path)
        assert command[-1] == str(tmp_path)
    else:
        assert command[-1] == str(tmp_path / "config.yaml")


@pytest.mark.parametrize("script", [adaptive, hints])
def test_fully_cached_sweep_does_not_start_ray(script, tmp_path, monkeypatch, fake_ray):
    launch = prepare_tasks(tmp_path, script, monkeypatch)
    for name in ("pending", "failed"):
        if script is adaptive:
            (tmp_path / name / "results.json").write_text("{}")
        else:
            for fraction in hints.FRACTIONS:
                (tmp_path / name / f"hints_{fraction:g}.json").write_text("{}")
    launch()
    assert fake_ray.calls == []
    assert json.loads((tmp_path / "failures.json").read_text()) == []

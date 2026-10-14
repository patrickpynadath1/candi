import copy
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from core.sweeps import nested_tuning as tuning


def spec(tmp_path, fixed=False):
    config = tmp_path / "base.yaml"
    OmegaConf.save(OmegaConf.create({"optimizer": {"lr": 0.1}}), config)
    training = {
        "trials": 2,
        "space": {"optimizer.lr": [0.1, 0.2]},
        "initial_points": [{"optimizer.lr": 0.1}, {"optimizer.lr": 0.2}],
    }
    if fixed:
        checkpoint = tmp_path / "fixed.ckpt"
        checkpoint.touch()
        training = {"trials": 1, "space": {}, "checkpoint": str(checkpoint)}
    return {
        "name": "example",
        "config": str(config),
        "training": training,
        "evaluation": {
            "tuning_indices": [0, 1],
            "test_indices": [2, 3],
            "seeds": [41],
            "test_seeds": [1041],
            "nfe": 4,
            "direction": "maximize",
            "score": {"kind": "metric", "metric": "accuracy"},
        },
        "inference": [
            {
                "name": "native",
                "sampler": {},
                "budget_parameter": "steps",
                "space": {"algorithm.predictor.temperature": [0.5, 1.0]},
                "trials": 2,
                "initial_points": [
                    {"algorithm.predictor.temperature": 0.5},
                    {"algorithm.predictor.temperature": 1.0},
                ],
            }
        ],
    }


@pytest.mark.parametrize("fixed", [False, True])
def test_same_nested_path_only_winner_sees_test_and_resume_is_noop(
    tmp_path, monkeypatch, fixed
):
    settings = spec(tmp_path, fixed)
    calls = []
    lrs = {}

    def execute(stage, payload, output):
        calls.append((stage, copy.deepcopy(payload)))
        if stage == "train":
            assert "test_indices" not in payload["validation"]
            path = str(output / "last.ckpt")
            lrs[path] = payload["parameters"]["optimizer.lr"]
            return {"config": settings["config"], "checkpoint": path}
        evaluation = payload["evaluation"]
        assert "test_indices" not in evaluation and "test_seeds" not in evaluation
        lr = lrs.get(payload["checkpoint"]["checkpoint"], 0.2)
        if evaluation["indices"] == [2, 3]:
            # Tests may rank candidates differently; they cannot affect selection.
            assert lr == 0.2
            assert payload["parameters"]["algorithm.predictor.temperature"] == 1.0
            assert evaluation["seeds"] == [1041]
            return {"score": -999.0}
        assert evaluation["indices"] == [0, 1]
        return {"score": lr + payload["parameters"]["algorithm.predictor.temperature"]}

    monkeypatch.setattr(tuning, "execute", execute)
    winner = tuning.run_case(settings, tmp_path / "runs", concurrency=1)
    assert winner["score"] == 1.2
    root = tmp_path / "runs/example"
    assert (root / "training/selected.json").exists()  # Also for inference-only.
    assert sum(stage == "train" for stage, _ in calls) == (0 if fixed else 2)
    test_calls = [
        p
        for stage, p in calls
        if stage == "evaluate" and p["evaluation"]["indices"] == [2, 3]
    ]
    assert len(test_calls) == 1
    count = len(calls)
    tuning.run_case(settings, tmp_path / "runs", concurrency=1)
    assert len(calls) == count
    settings["evaluation"]["test_indices"] = [8, 9]
    with pytest.raises(ValueError, match="Protocol changed"):
        tuning.run_case(settings, tmp_path / "runs")


def test_failed_inner_search_does_not_touch_test_or_retrain_on_resume(
    tmp_path, monkeypatch
):
    settings = spec(tmp_path)
    settings["training"]["trials"] = 1
    calls = []
    fail = True

    def execute(stage, payload, output):
        nonlocal fail
        calls.append(stage)
        if stage == "train":
            return {
                "config": settings["config"],
                "checkpoint": str(output / "last.ckpt"),
            }
        if fail:
            fail = False
            raise RuntimeError("transient evaluation failure")
        return {"score": 1.0}

    monkeypatch.setattr(tuning, "execute", execute)
    with pytest.raises(RuntimeError, match="transient"):
        tuning.run_case(settings, tmp_path / "runs", 1)
    assert not (tmp_path / "runs/example/final/result.json").exists()
    tuning.run_case(settings, tmp_path / "runs", 1)
    assert calls.count("train") == 1


def test_validation_adapter_restricts_loss_and_generation():
    from core.tasks.base import PrimaryMetric

    class Task:
        primary_metric = PrimaryMetric("accuracy", "max")

        def evaluation_dataset(self, name):
            return list(range(10))

        def training_dataset(self):
            return [99]

    task = tuning.ValidationTask(Task(), [1, 4])
    assert list(task.validation_dataset()) == [1, 4]
    assert list(task.evaluation_dataset("validation")) == [1, 4]
    assert task.training_dataset() == [99]
    with pytest.raises(ValueError):
        task.evaluation_dataset("test")


def test_real_training_keeps_periodic_callback_with_validation_partition(tmp_path):
    from core.tests.test_programmatic_sweeps import tiny

    config = tiny()
    config["trainer"].update(limit_val_batches=1, val_check_interval=1)
    config["callback"].update(every_n_steps=1, subset_size=4)
    config["callback"]["output_dir"] = str(tmp_path / "old-run-callback")
    output = tmp_path / "training"
    checkpoint = tuning._train_worker(
        {
            "config": config,
            "parameters": {"optimizer.lr": 0.001},
            "validation": {"indices": [1], "seeds": [41]},
            "resources": {"gpu": 0},
        },
        output,
    )
    assert Path(checkpoint["checkpoint"]).is_file()
    reports = list(output.glob("task_callback/step_*.json"))
    assert len(reports) == 1
    assert not (tmp_path / "old-run-callback").exists()
    assert json.loads(reports[0].read_text())["num_total"] == 1
    trained_config = OmegaConf.load(output / "config.yaml")
    assert trained_config.task.indices == [1]
    evaluation_config = OmegaConf.load(checkpoint["config"])
    assert evaluation_config.task._target_.endswith("TinyTask")
    result = tuning._evaluate_worker(
        {
            "checkpoint": checkpoint,
            "evaluation": {
                "indices": [2, 3],
                "seeds": [42],
                "nfe": 3,
                "score": {"metric": "accuracy"},
            },
            "policy": {
                "sampler": {
                    "_target_": "core.samplers.ancestral.AncestralSampler",
                    "steps": 2,
                },
                "budget_parameter": "steps",
            },
            "parameters": {},
            "resources": {"gpu": 0},
        },
        tmp_path / "evaluation",
    )
    assert 0 <= result["score"] <= 1


def test_duplicate_parallel_candidates_share_artifacts(tmp_path):
    calls = []

    def objective(parameters, output):
        calls.append(parameters)
        return {"score": 1.0}

    tuning.search(
        tmp_path / "search",
        {"space": {"x": [1]}, "trials": 2},
        "maximize",
        objective,
        2,
    )
    assert len(calls) == 1


def test_existing_checkpoint_cannot_claim_training_search(tmp_path):
    settings = spec(tmp_path, fixed=True)
    settings["training"]["space"] = {"optimizer.lr": [0.1, 0.2]}
    with pytest.raises(ValueError, match="fixed training candidate"):
        tuning.validate(settings)


def test_selection_cannot_use_actual_test_dataset(tmp_path):
    settings = spec(tmp_path, fixed=True)
    settings["evaluation"]["dataset"] = "test"
    with pytest.raises(ValueError, match="validation dataset"):
        tuning.validate(settings)


def test_fixed_checkpoint_workers_use_pinned_config(tmp_path, monkeypatch):
    settings = spec(tmp_path, fixed=True)
    original = Path(settings["config"])
    paths = []

    def execute(stage, payload, output):
        path = Path(payload["checkpoint"]["config"])
        assert path != original
        assert OmegaConf.load(path).optimizer.lr == 0.1
        original.write_text("optimizer: {lr: 999}\n")
        paths.append(path)
        return {"score": 1.0}

    monkeypatch.setattr(tuning, "execute", execute)
    tuning.run_case(settings, tmp_path / "runs", 1)
    assert len(paths) == 3


def test_sqlite_trial_lifecycle_is_serial_while_evaluations_are_parallel(
    tmp_path, monkeypatch
):
    import threading
    import optuna

    driver = threading.get_ident()
    barrier = threading.Barrier(4)
    calls = []
    for method in ("ask", "tell"):
        original = getattr(optuna.study.Study, method)

        def checked(self, *args, _original=original, **kwargs):
            assert threading.get_ident() == driver
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(optuna.study.Study, method, checked)

    def objective(parameters, output):
        assert threading.get_ident() != driver
        barrier.wait(timeout=10)
        calls.append(parameters["x"])
        return {"score": float(parameters["x"])}

    settings = {
        "space": {"x": list(range(8))},
        "initial_points": [{"x": x} for x in range(8)],
        "trials": 8,
    }
    winner = tuning.search(tmp_path, settings, "maximize", objective, 4)
    assert winner["score"] == 7
    assert sorted(calls) == list(range(8))
    study = optuna.load_study(
        study_name="search", storage=f"sqlite:///{tmp_path / 'optuna.sqlite'}"
    )
    assert len(study.trials) == 8
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    tuning.search(tmp_path, settings, "maximize", objective, 4)
    assert len(calls) == 8


def test_previous_case_entrypoint_remains_compatible():
    assert tuning.run_study is tuning.run_case


@pytest.mark.parametrize("flag", ["--case-concurrency", "--study-concurrency"])
def test_case_concurrency_cli_accepts_both_names(tmp_path, monkeypatch, flag):
    import sys
    import ray

    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]")
    connections = []
    monkeypatch.setattr(ray, "init", lambda **kwargs: connections.append(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        ["nested_tuning", str(manifest), "--output", str(tmp_path), flag, "2"],
    )
    tuning.main()
    assert connections == [{"address": "auto"}]

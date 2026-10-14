"""Method identity, task isolation, and real train/evaluate provenance."""

import json
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest

from examples.task_battery import build_jobs
from core.sweeps.configuration import ROOT, compose
from core.sweeps.jobs import InferenceJob
from core.sweeps.queue import _inference_config
from core.training.runner import run


def definitions():
    return (
        OmegaConf.to_container(
            OmegaConf.load(ROOT / "configs/algorithm_definition/masked.yaml")
        ),
        OmegaConf.to_container(
            OmegaConf.load(ROOT / "configs/battery/small_tasks.yaml")
        ),
    )


def small_definitions(tmp_path):
    algorithm, battery = definitions()
    algorithm["settings"].update(
        {
            "model.backbone.hidden_size": 16,
            "model.backbone.cond_dim": 8,
            "model.backbone.n_blocks": 1,
            "model.backbone.n_heads": 2,
            "sampler.steps": 1,
        }
    )
    algorithm["policies"] = [{"name": "native", "overrides": {"sampler.steps": 1}}]
    battery["seeds"] = [1]
    battery["settings"].update(
        {
            "trainer.accelerator": "cpu",
            "trainer.precision": "32-true",
            "trainer.max_steps": 2,
            "trainer.val_check_interval": 2,
            "trainer.log_every_n_steps": 1,
            "loader.num_workers": 0,
            "loader.global_batch_size": 2,
            "loader.eval_batch_size": 2,
            "checkpoint.every_n_steps": 2,
            "callback.every_n_steps": 2,
            "callback.subset_size": 2,
            "evaluation.device": "cpu",
            "evaluation.max_examples": 2,
            "training.warmup_steps": 0,
        }
    )
    battery["tasks"] = [
        {
            "name": "queens",
            "task": "nqueens",
            "settings": {
                "data.cache_dir": str(tmp_path / "data"),
                "data.num_train": 12,
                "data.num_valid": 4,
            },
            "evaluations": [
                {"name": "conditional", "overrides": {}},
                {
                    "name": "unconditional",
                    "overrides": {"task.evaluation_mode": "unconditional"},
                },
            ],
        }
    ]
    return algorithm, battery


def test_five_tasks_one_algorithm_and_policy_construction():
    a, b = definitions()
    jobs = build_jobs(a, b)
    algorithm_id = jobs[0].config["benchmark"]["algorithm_id"]
    assert len(jobs) == 25
    assert sum(len(j.inference) for j in jobs) == 90
    assert {j.config["benchmark"]["algorithm_id"] for j in jobs} == {algorithm_id}
    assert {j.config["task"]["_target_"] for j in jobs} == {
        "core.tasks.sudoku.SudokuTask",
        "core.tasks.nqueens.NQueensTask",
        "core.tasks.text8.Text8Task",
        "core.tasks.tinystories.TinyStoriesTask",
        "core.tasks.maze.MazeTask",
    }
    for choice in jobs[0].inference:
        sampler = (
            compose(overrides={"sampler": choice.sampler}).sampler
            if choice.sampler
            else OmegaConf.create(jobs[0].config).sampler
        )
        for key, value in choice.overrides.items():
            if key.startswith("sampler."):
                OmegaConf.update(sampler, key.removeprefix("sampler."), value)
        instantiate(sampler)


def test_identity_changes_with_method_but_not_task_budget(tmp_path):
    a, b = small_definitions(tmp_path)
    first = build_jobs(a, b)[0].config["benchmark"]["algorithm_id"]
    b["settings"]["trainer.max_steps"] = 5
    assert build_jobs(a, b)[0].config["benchmark"]["algorithm_id"] == first
    a["settings"]["self_conditioning.enabled"] = True
    assert build_jobs(a, b)[0].config["benchmark"]["algorithm_id"] != first


@pytest.mark.parametrize(
    "side,key",
    [
        ("algorithm", "task.length"),
        ("algorithm", "trainer.max_steps"),
        ("battery", "model.backbone.hidden_size"),
        ("battery", "sampler.steps"),
    ],
)
def test_definition_boundaries(tmp_path, side, key):
    a, b = small_definitions(tmp_path)
    (a if side == "algorithm" else b)["settings"][key] = 12
    with pytest.raises(ValueError, match="outside"):
        build_jobs(a, b)


def test_inference_cannot_modify_training_data():
    with pytest.raises(ValueError):
        InferenceJob(overrides={"data.num_train": 2})
    with pytest.raises(ValueError):
        InferenceJob(overrides={"task.length": 12})


@pytest.mark.parametrize("task", ["nqueens", "maze"])
@pytest.mark.parametrize("corruption", ["masked", "gaussian"])
def test_cpu_training_and_both_evaluations_link_to_definition(
    tmp_path, task, corruption
):
    a, b = small_definitions(tmp_path)
    a["components"]["hybrid_view@model.views.content"] = corruption
    if task == "maze":
        pytest.importorskip("maze_dataset")
        b["tasks"] = [
            {
                "name": "maze",
                "task": "maze",
                "settings": {
                    "data.cache_dir": str(tmp_path / "data"),
                    "data.grid_size": 4,
                    "data.num_train": 8,
                    "data.num_valid": 4,
                    "data.num_test": 4,
                    "data.max_path_edges": 24,
                    "task.vocab_size": 48,
                    "task.length": 64,
                },
                "evaluations": [
                    {"name": "validation", "overrides": {}},
                    {
                        "name": "test",
                        "overrides": {"evaluator.evaluation_name": "test"},
                    },
                ],
            }
        ]
    jobs = build_jobs(a, b)
    algorithm_id = jobs[0].config["benchmark"]["algorithm_id"]
    directory = tmp_path / "battery"
    cfg = OmegaConf.create(jobs[0].config)
    cfg.output_dir = str(directory / "training/run")
    run(cfg)
    periodic = json.loads(
        (Path(cfg.output_dir) / "task_callback/step_00000002.json").read_text()
    )
    assert periodic["benchmark"]["algorithm_id"] == algorithm_id
    for i, choice in enumerate(jobs[0].inference):
        ev = OmegaConf.create(
            _inference_config(Path(cfg.output_dir), choice, "last.ckpt")
        )
        ev.output_dir = str(directory / f"evaluation-{i}/run")
        run(ev)
        result = json.loads((Path(ev.output_dir) / "results.json").read_text())
        assert result["benchmark"]["algorithm_id"] == algorithm_id
        saved = OmegaConf.load(Path(ev.output_dir) / "config.yaml")
        assert OmegaConf.to_container(saved.algorithm_definition) == {
            key: a[key] for key in ("components", "settings", "policies")
        }


def test_preview_no_ray_or_data(monkeypatch, tmp_path, capsys):
    from examples.task_battery import main
    import ray

    monkeypatch.setattr(ray, "init", lambda **kw: pytest.fail("preview launched Ray"))
    main([])
    assert "Preview only" in capsys.readouterr().out


def test_battery_cannot_change_periodic_sampler(tmp_path):
    a, b = small_definitions(tmp_path)
    b["settings"]["callback.sampler_config"] = {"_target_": "foreign.Sampler"}
    with pytest.raises(ValueError, match="sampler definitions"):
        build_jobs(a, b)

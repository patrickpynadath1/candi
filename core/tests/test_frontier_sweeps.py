import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from core.training.runner import run as train
from core.sweeps.frontier import pareto_flags, run
from core.tests.test_mdlm import config


def test_pareto_directions_ties_and_grouping():
    rows = [
        dict(cost=2, quality=1, seed=1),
        dict(cost=1, quality=1, seed=1),
        dict(cost=1, quality=1, seed=1),
        dict(cost=3, quality=2, seed=1),
        dict(cost=0, quality=3, seed=2),
    ]
    assert pareto_flags(rows, {"cost": "min", "quality": "max"}, ["seed"]) == [
        False,
        True,
        True,
        True,
        True,
    ]
    assert pareto_flags(rows, {"cost": "min", "quality": "max"}) == [
        False,
        False,
        False,
        False,
        True,
    ]
    for invalid in (None, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite scalar"):
            pareto_flags(
                [dict(cost=invalid, quality=1)], {"cost": "min", "quality": "max"}
            )


@pytest.mark.parametrize("preset", ["sudoku", "text8", "openwebtext"])
def test_frontier_presets_compose(preset):
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[1] / "new_configs")
    ):
        overrides = [f"frontier={preset}", "output_dir=/tmp/frontier_test"]
        if preset != "sudoku":
            overrides += [f"+experiment={preset}", "sampler=ancestral"]
        settings = compose(config_name="frontier", overrides=overrides)
    assert settings.mode == "evaluate"
    for axis in settings.frontier.axes:
        assert OmegaConf.select(settings, axis.parameter) is not None


def test_real_checkpoint_sweep_resume_and_dry_run(tmp_path, monkeypatch):
    settings = config()
    settings.task = OmegaConf.create(
        {
            "_target_": "core.tests.test_entrypoint.TinyTask",
            "vocab_size": 5,
            "length": 8,
        }
    )
    settings.output_dir = str(tmp_path / "train")
    settings.trainer.enable_progress_bar = False
    settings.callback.every_n_steps = 0
    train(settings)
    settings.mode = "evaluate"
    settings.evaluation.checkpoint = str(tmp_path / "train/checkpoints/last.ckpt")
    settings.output_dir = str(tmp_path / "sweep")
    OmegaConf.set_struct(settings, False)
    settings.frontier = OmegaConf.create(
        {
            "axes": [
                {"parameter": "parameterization.temperature", "values": [0.8, 1.0]}
            ],
            "seeds": [1],
            "objectives": {"mean_nfe": "min", "accuracy": "max"},
            "group_by": ["seed"],
            "resume": True,
            "dry_run": True,
        }
    )
    assert (
        run(settings, evaluation_entrypoint=Path(__file__).parents[2] / "main.py") == []
    )
    manifest = json.loads((tmp_path / "sweep/manifest.json").read_text())
    assert len(manifest["points"]) == 2
    assert not list((tmp_path / "sweep").glob("points/*/results.json"))
    settings.frontier.dry_run = False
    rows = run(settings, evaluation_entrypoint=Path(__file__).parents[2] / "main.py")
    assert len(rows) == 2
    assert all(row["num_total"] == 2 for row in rows)
    summary = json.loads((tmp_path / "sweep/frontiers.json").read_text())
    assert len(summary["points"]) == 2
    assert (tmp_path / "sweep/frontier.svg").exists()

    def fail(*args, **kwargs):
        raise AssertionError("completed evaluations should be reused")

    monkeypatch.setattr("core.sweeps.frontier.subprocess.run", fail)
    assert (
        run(settings, evaluation_entrypoint=Path(__file__).parents[2] / "main.py")
        == rows
    )
    settings.evaluation.max_examples = 1
    settings.frontier.dry_run = True
    run(settings, evaluation_entrypoint=Path(__file__).parents[2] / "main.py")
    changed = json.loads((tmp_path / "sweep/manifest.json").read_text())
    assert {p["point_id"] for p in changed["points"]}.isdisjoint(
        p["point_id"] for p in manifest["points"]
    )
    settings.frontier.axes[0].parameter = "sampler.nonexistent"
    with pytest.raises(ValueError, match="unknown sweep parameter"):
        run(settings, evaluation_entrypoint=Path(__file__).parents[2] / "main.py")

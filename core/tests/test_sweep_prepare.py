"""Exercise the actual Hydra CLI rather than reimplementing its grid expansion."""

import json
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
import pytest
from hydra.utils import instantiate

from core.sweeps.configuration import ROOT, compose
from core.sweeps.prepare import collect, prepare_case


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / "reference.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "text8-gaussian-rank-linear-cdcd-sc-off_20m",
                    "evaluation": {
                        "tuning_indices": [0, 1],
                        "test_indices": [2, 3],
                        "score": {"kind": "metric", "name": "score"},
                        "direction": "maximize",
                    },
                }
            ]
        )
    )
    return path


def run_prepare(output, reference, *overrides):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "core.sweeps.prepare",
            "--multirun",
            "+sweep=hybrids",
            "task=battery/text8",
            "gaussian_state=one_hot",
            "path_schedule=adaptive",
            "sc=off",
            f"preparation.output={output}",
            f"preparation.reference_manifest={reference}",
            *overrides,
        ],
        cwd=ROOT,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_native_hydra_grid_and_manifest(tmp_path, reference):
    output = tmp_path / "export"
    result = run_prepare(output, reference)
    assert result.returncode == 0, result.stdout + result.stderr
    assert collect(output) == 2
    cases = json.loads((output / "manifest.json").read_text())
    jobs = OmegaConf.to_container(OmegaConf.load(output / "training-jobs.yaml"))
    assert {c["name"] for c in cases} == {
        f"text8-gaussian-one_hot-{view}-adaptive-cross_entropy-sc-off-seed-1"
        for view in ["masked", "uniform"]
    }
    for case, job in zip(cases, jobs):
        assert case["name"] == job["name"]
        assert case["evaluation"]["tuning_indices"] == [0, 1]
        assert case["evaluation"]["test_indices"] == [2, 3]
        config = job["config"]
        assert config["trainer"]["max_steps"] == 20000
        assert config["trainer"]["precision"] == "bf16-mixed"
        assert config["model"]["backbone"]["n_blocks"] == 8
        assert config["model"]["views"]["content"]["time_map"]
        assert not config["self_conditioning"]["enabled"]
        assert "difficulty" not in config["data"]
        assert "preparation" not in config
        assert config["sampler"]["steps"] == 127
        assert [p["name"] for p in case["inference"]] == [
            "native",
            "native_greedy",
            "eb_greedy",
        ]
        assert case["inference"][1]["sampler"]["greedy_clean_estimate"] is True
        assert Path(case["config"]).is_file()
    before = (output / "configs" / f"{cases[0]['name']}.yaml").read_bytes()
    repeated = run_prepare(output, reference)
    assert repeated.returncode != 0
    assert "FileExistsError" in repeated.stderr
    assert before == (output / "configs" / f"{cases[0]['name']}.yaml").read_bytes()


@pytest.mark.parametrize("name", ["../escape", "", ".", "/absolute"])
def test_invalid_case_name(name):
    with pytest.raises(ValueError, match="Invalid job name"):
        prepare_case(OmegaConf.create({"preparation": {"case_name": name}}))


def test_collect_rejects_incomplete_export(tmp_path):
    folder = tmp_path / "cases" / "incomplete"
    folder.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        collect(tmp_path)
    assert not (tmp_path / "manifest.json").exists()


def test_blurring_defaults_are_composed_before_export():
    config = compose(overrides=["+sweep=continuous", "content_view=blurring"])
    view = config.model.views.content
    assert "defaults" not in view
    assert view.schedule._target_ == "core.noise_schedules.blurring.BlurringSchedule"
    instance = instantiate(view)
    assert type(instance.schedule).__name__ == "BlurringSchedule"

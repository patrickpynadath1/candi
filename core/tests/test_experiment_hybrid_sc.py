"""experiments/hybrid-sc: the sweep expands to the intended 48-case grid of two-view models."""

import json
import os

from omegaconf import OmegaConf

from core.tests.helpers import ROOT, run_python

VIEWS = ("masked", "uniform")
PATHS = ("gaussian_embedding", "simplex", "spherical", "spherical_vmf")
SC = ("none", "expected_embedding")
TASKS = ("sudoku_hard", "text8", "tinystories-bpe-10000")


def test_sweep_expands_to_views_x_paths_x_sc_x_tasks(tmp_path):
    plan = tmp_path / "plan"
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    run_python("-m", "core.sweeps.prepare", "-m", "+sweep=hybrid_sc",
               f"preparation.output={plan}", env=env)
    run_python("-m", "core.sweeps.prepare", "collect", str(plan), env=env)
    cases = json.loads((plan / "manifest.json").read_text())
    assert {case["name"] for case in cases} == {
        f"{task}-{view}-{path}-{sc}-seed-1"
        for task in TASKS for view in VIEWS for path in PATHS for sc in SC
    }
    for case in cases:
        config = OmegaConf.load(case["config"])
        # Every case is a two-view model, continuous path first, discrete view second ...
        assert list(config.model.views) == ["content", "discrete"]
        # ... and CDCD learns the clock of the continuous view (index 0).
        assert config.time_sampler._target_ == "core.time_sampling.cdcd.CDCDTimeSampler"
        assert list(config.algorithm.adaptive_time_views) == [0]
        assert case["evaluation"]["nfe"] == 128

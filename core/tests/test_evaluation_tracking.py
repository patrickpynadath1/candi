from pathlib import Path
from types import SimpleNamespace
import sys

from omegaconf import OmegaConf

from core.training.evaluation_tracking import log_evaluation


def test_evaluation_tracking_preserves_project_and_distinguishes_sampler(monkeypatch):
    calls, logged, summary = [], [], {}

    class Run:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def log(self, metrics):
            logged.append(metrics)

    run = Run()
    run.summary = summary

    def init(**kwargs):
        calls.append(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    config = OmegaConf.create(
        {
            "wandb": {
                "project": "scCANDI",
                "entity": "rz-text-dlp",
                "name": "pbmc3k-masked",
                "group": "sweep",
                "job_type": "train",
            },
            "sampler": {"steps": 32},
        }
    )
    result = {
        "mmd": 0.2,
        "mean_nfe": 33,
        "per_cell_type": {"B": {"mmd": 0.1}},
        "records": [{"generated": [0, 1]}],
    }
    log_evaluation(config, result, Path("/tmp/eval"))
    config.sampler.steps = 180
    log_evaluation(config, result, Path("/tmp/eval"))
    assert calls[0]["project"] == "scCANDI"
    assert calls[0]["entity"] == "rz-text-dlp"
    assert calls[0]["job_type"] == "evaluate"
    assert calls[0]["name"] != calls[1]["name"]
    assert logged[0] == {"mmd": 0.2, "mean_nfe": 33, "per_cell_type/B/mmd": 0.1}
    assert summary["parent_training_name"] == "pbmc3k-masked"
    assert config.wandb.job_type == "train"

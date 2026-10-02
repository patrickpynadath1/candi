"""The same infrastructure must work with another Task selected by config."""

import json
from dataclasses import dataclass

from hydra.utils import instantiate
from hydra.errors import InstantiationException
from omegaconf import OmegaConf
import pytest
import torch
from torch.utils.data import default_collate

from core.training.runner import run
from core.tasks import Task, PrimaryMetric
from core.tests.test_mdlm import config


class TinyTask(Task):
    name = "tiny"
    primary_metric = PrimaryMetric("accuracy", "max")

    def __init__(self, vocab_size, length):
        self.vocab_size, self.length = vocab_size, length

    def prepare_data(self):
        pass

    def training_dataset(self):
        return [
            {
                "input_ids": torch.arange(self.length) % self.vocab_size,
                "loss_mask": torch.ones(self.length, dtype=torch.bool),
            }
            for _ in range(4)
        ]

    def validation_dataset(self):
        return self.training_dataset()

    def collate_training(self, examples):
        return default_collate(examples)

    def prepare_generation(self, examples):
        shape = (len(examples), self.length)
        return torch.zeros(shape, dtype=torch.long), torch.zeros(
            shape, dtype=torch.bool
        )

    def create_metric_accumulator(self, evaluation_name):
        return TinyMetrics()


@dataclass(frozen=True)
class TinyRecord:
    index: int
    exact: bool


@dataclass(frozen=True)
class TinyAggregate:
    num_total: int
    accuracy: float
    records: list[TinyRecord]


class TinyMetrics:
    def __init__(self):
        self.records = []

    def update(self, samples, examples, indices):
        for sample, example, index in zip(samples, examples, indices):
            assert sample.shape == (8,)
            self.records.append(
                TinyRecord(index, bool(torch.equal(sample, example["input_ids"])))
            )

    def compute(self):
        return TinyAggregate(
            num_total=len(self.records),
            accuracy=sum(r.exact for r in self.records) / len(self.records),
            records=list(self.records),
        )


def test_entrypoint_trains_and_evaluates_configured_non_sudoku_task(
    tmp_path, monkeypatch
):
    settings = config()
    settings.task = OmegaConf.create(
        {
            "_target_": "core.tests.test_entrypoint.TinyTask",
            "vocab_size": 5,
            "length": 8,
        }
    )
    settings.callback.every_n_steps = 2
    settings.output_dir = str(tmp_path / "train")
    settings.trainer.enable_progress_bar = False
    run(settings)
    checkpoint = tmp_path / "train/checkpoints/last.ckpt"
    assert checkpoint.exists()
    assert (tmp_path / "train/task_callback/step_00000002.json").exists()

    def forbid_training_preparation(self):
        raise AssertionError("standalone evaluation must not prepare training data")

    monkeypatch.setattr(TinyTask, "prepare_data", forbid_training_preparation)
    settings.mode = "evaluate"
    settings.evaluation.checkpoint = str(checkpoint)
    settings.output_dir = str(tmp_path / "evaluate")
    settings.evaluation.max_examples = None
    run(settings)
    result = json.loads((tmp_path / "evaluate/results.json").read_text())
    assert result["primary_metric"] == {
        "name": "accuracy",
        "mode": "max",
        "dataset": "validation",
    }
    assert result["num_total"] == 4


def test_sudoku_validation_belongs_to_selected_task():
    with pytest.raises(InstantiationException, match="SudokuTask requires"):
        instantiate(config("task.length=8").task)

import json
import random
from types import SimpleNamespace

import datasets
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from samplers import SamplingResult
from tasks import reasoning_callback as module
from tasks.reasoning import encode_record, get_reasoning_tokenizer


def setup(monkeypatch, tmp_path, task="zebra", full=False):
    tokenizer = get_reasoning_tokenizer(task)
    if task == "zebra":
        record = dict(
            task=task,
            puzzle=["=", "LHS", "c", "0", "0", "RHS", "n", "0", "0", "CLUE_END"],
            answer=[[0, 1, 2], [1, 2, 0], [2, 0, 1]],
            attributes=3,
            houses=3,
        )
        prompt, length = 32, 44
    else:
        answer = [[(r * 3 + r // 3 + c) % 9 + 1 for c in range(9)] for r in range(9)]
        record = dict(task=task, puzzle=[[0] * 9 for _ in range(9)], answer=answer)
        prompt, length = 91, 180
    record.update(source_id="example", group_id="group")
    encoded = encode_record(record, tokenizer, prompt, length)
    dataset = datasets.Dataset.from_list([encoded] * 4).with_format(
        "torch",
        columns=["input_ids", "attention_mask", "prompt_mask"],
        output_all_columns=True,
    )
    manifest = dict(full_training=full, fingerprint="source-v1")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    splits = []

    def load(config, tokenizer, split):
        splits.append(split)
        return dataset

    monkeypatch.setattr(module, "load_prepared", load)

    class Model:
        device = torch.device("cpu")
        config = OmegaConf.create(
            {"data": {"train": task, "prepared_dir": str(tmp_path)}}
        )
        in_eval = False

        def _eval_mode(self):
            self.in_eval = True

        def _train_mode(self):
            self.in_eval = False

        def generate_samples(self, count, **kwargs):
            assert self.in_eval
            mask = kwargs["prompt_mask"]
            assert (kwargs["prompt_tokens"][~mask] == 0).all()
            torch.rand(3)
            np.random.rand()
            random.random()
            return SamplingResult(
                torch.tensor([encoded["input_ids"]] * count),
                {"nfe": kwargs["num_steps"]},
            )

    model = Model()
    model.tokenizer = tokenizer
    barriers = []
    trainer = SimpleNamespace(
        global_step=1000,
        is_global_zero=True,
        logger=None,
        strategy=SimpleNamespace(barrier=barriers.append),
    )
    callback = module.ReasoningEvaluationCallback(
        subset_size=2, num_steps=4, output_dir=tmp_path / "monitor"
    )
    return callback, model, trainer, splits, barriers


@pytest.mark.parametrize(
    "task,full,expected",
    [("zebra", False, "validation"), ("sudoku_extreme", True, "test")],
)
def test_monitor_scores_and_preserves_rng(monkeypatch, tmp_path, task, full, expected):
    callback, model, trainer, splits, barriers = setup(
        monkeypatch, tmp_path, task, full
    )
    torch_state = torch.get_rng_state().clone()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    callback.on_train_batch_end(trainer, model, None, None, 0)
    result = json.loads((tmp_path / "monitor/step_00001000.json").read_text())
    assert result["solve_rate"] == 1 and result["num_total"] == 2
    assert result["monitoring_only"] and not result["checkpoint_selection"]
    assert splits == [expected] and len(barriers) == 1
    assert not model.in_eval and torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    callback.on_train_batch_end(trainer, model, None, None, 1)
    assert len(barriers) == 1  # repeated accumulation batch at same optimizer step
    restored = module.ReasoningEvaluationCallback(output_dir=tmp_path / "other")
    restored.load_state_dict(callback.state_dict())
    restored.on_train_batch_end(trainer, model, None, None, 2)
    assert len(barriers) == 1
    trainer.global_step = 2000
    callback.on_train_batch_end(trainer, model, None, None, 3)
    assert len(barriers) == 2 and splits == [expected]


def test_monitor_failure_restores_training_and_rng(monkeypatch, tmp_path):
    callback, model, trainer, _, _ = setup(monkeypatch, tmp_path)

    def fail(*args, **kwargs):
        torch.rand(3)
        raise RuntimeError("sampling failed")

    model.generate_samples = fail
    state = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="sampling failed"):
        callback.on_train_batch_end(trainer, model, None, None, 0)
    assert not model.in_eval and torch.equal(state, torch.get_rng_state())
    assert callback._last_step == -1


def test_nonzero_rank_only_joins_barrier(monkeypatch, tmp_path):
    callback, model, trainer, splits, barriers = setup(monkeypatch, tmp_path)
    trainer.is_global_zero = False
    callback.on_train_batch_end(trainer, model, None, None, 0)
    assert not splits and len(barriers) == 1

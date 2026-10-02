"""Offline fixtures exercise TinyGSM data, variable prompts and generation scoring."""

from contextlib import nullcontext
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from core.tasks.tinygsm import TinyGSMTask
from core.tasks.tinygsm_metrics import TinyGSMMetrics, TinyGSMAggregate
from core.tasks.metrics.gsm8k import numeric_answer, gold_answer, numbers_equal
from core.tasks.sandbox import execute_python, bubblewrap_command
from core.evaluation.generation import evaluate


class Tokenizer:
    bos_token_id, eos_token_id, pad_token_id = 0, 1, 2

    def __len__(self):
        return 131

    def get_vocab(self):
        return {chr(i): i + 3 for i in range(128)}

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) + 3 for c in text]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i - 3) for i in ids if i >= 3)


def make_task(tmp_path, **kwargs):
    training = tmp_path / "training.json"
    training.write_text(
        json.dumps([{"question": f"q{i}", "code": "answer: 42"} for i in range(10)])
    )
    evaluation = tmp_path / "gsm.json"
    evaluation.write_text(
        json.dumps(
            [
                {"prompt": "long question", "response_ground_truth": "#### 42"},
                {"prompt": "q", "response_ground_truth": "#### 42"},
            ]
        )
    )
    data = OmegaConf.create(
        dict(
            cache_dir=str(tmp_path / "cache"),
            data_path=str(training),
            tokenizer_name_or_path="fixture",
            separator="\n",
            wrap=False,
            streaming=False,
            train_on_prompt=False,
            train_on_pad=True,
            filter_too_long=True,
            max_examples=None,
            val_ratio=0.2,
            val_seed=42,
        )
    )
    scoring = OmegaConf.create(
        dict(
            cache_dir=str(tmp_path),
            data_path=str(evaluation),
            separator="\n",
            scoring="numeric",
            timeout=0.5,
            bootstrap_size=20,
            seed=1,
        )
    )
    for k, v in kwargs.items():
        data[k] = v
    return TinyGSMTask(data, scoring, vocab_size=131, length=48, tokenizer=Tokenizer())


@pytest.mark.parametrize("train_on_pad", [False, True])
def test_training_masks_and_cache(tmp_path, train_on_pad):
    task = make_task(tmp_path, train_on_pad=train_on_pad)
    task.prepare_data()
    examples = [task.training_dataset()[0]]
    batch = task.collate_training(examples)
    assert batch["input_ids"].shape == (1, 48)
    assert not batch["loss_mask"][0, :4].any()
    assert batch["loss_mask"][0, 4:15].all()
    assert batch["loss_mask"][0, 15:].eq(train_on_pad).all()
    assert torch.equal(batch["loss_mask"], batch["corruption_mask"])
    assert batch["padding_mask"][0, 15:].all()
    assert len(task.validation_dataset()) == 2


def test_generation_evaluator_and_exact_length_batches(tmp_path):
    task = make_task(tmp_path)
    dataset = task.evaluation_dataset("gsm8k")
    assert list(task.evaluation_batches(dataset, [0, 1], 2)) == [[1], [0]]

    class Algorithm:
        def generate(self, sampler, prompt, mask):
            output = prompt.clone()
            for row in range(len(output)):
                start = int(mask[row].sum())
                response = [ord(c) + 3 for c in "answer: 42"] + [1]
                output[row, start : start + len(response)] = torch.tensor(response)
            return output, {"nfe_per_sample": [int(m.sum()) for m in mask]}

    module = SimpleNamespace(
        device=torch.device("cpu"),
        algorithm=Algorithm(),
        evaluation_weights=lambda _: nullcontext(),
    )
    with patch.object(
        task, "prepare_data", side_effect=AssertionError("training must not load")
    ):
        result = evaluate(
            module,
            task,
            {"_target_": "core.samplers.ancestral.AncestralSampler"},
            batch_size=2,
        )
    assert result["accuracy"] == 1
    assert result["truncation_rate"] == 0
    assert [r["index"] for r in result["records"]] == [1, 0]
    assert all(r["nfe"] == r["prompt_tokens"] for r in result["records"])
    examples = [dataset[0], dataset[1]]
    prompt, mask = task.prepare_generation(examples)
    assert prompt[~mask].eq(0).all()
    assert mask[0].sum() != mask[1].sum()


def test_original_number_math_and_bootstrap():
    assert numeric_answer("answer: -$1,234.50") == -1234.5
    assert gold_answer("reasoning\n#### 42") == 42
    assert numbers_equal(0.333, 0.3334)
    metrics = TinyGSMMetrics(Tokenizer(), scoring="numeric", bootstrap_size=20, seed=1)
    example = {
        "input_ids": torch.tensor([0]),
        "prompt": "q",
        "response_ground_truth": "#### 42",
    }
    for i, answer in enumerate(["42", "41", "42"]):
        metrics.update(
            torch.tensor([[0] + [ord(c) + 3 for c in answer] + [1]]), [example], [i]
        )
    result = metrics.compute()
    assert isinstance(result, TinyGSMAggregate)
    assert result.accuracy == 2 / 3
    estimates = np.array([1.0, 0.0, 1.0])[
        np.random.default_rng(1).integers(0, 3, size=(20, 3))
    ].mean(1)
    assert result.accuracy_se == float(estimates.std(ddof=1))


def test_sandbox_fails_closed_and_checks_syntax():
    with (
        patch("core.tasks.sandbox.shutil.which", return_value=None),
        patch("core.tasks.sandbox.subprocess.run") as run,
    ):
        result = execute_python("def solve():\n return 42", entrypoint="solve")
        assert result.sandbox_error and not result.executed
        run.assert_not_called()
    for code in ["def solve(:", "import os", "def solve():\n return (1).__class__"]:
        assert not execute_python(code, entrypoint="solve").parsed
    with patch("core.tasks.sandbox.shutil.which", return_value="/usr/bin/bwrap"):
        command = bubblewrap_command()
    assert "--unshare-net" in command and "/home" not in command


def test_real_bubblewrap_arithmetic_and_timeout():
    result = execute_python(
        "def solve():\n import math\n return math.sqrt(81)", entrypoint="solve"
    )
    if result.sandbox_error:
        pytest.skip("host does not permit bubblewrap namespaces")
    assert result.executed and result.result == 9
    result = execute_python(
        "def solve():\n while True:\n  pass", entrypoint="solve", timeout=0.2
    )
    assert result.timed_out and not result.executed


def test_broken_sandbox_protocol_and_resource_status():
    with (
        patch("core.tasks.sandbox.bubblewrap_command", return_value=["bwrap"]),
        patch("core.tasks.sandbox.subprocess.run") as run,
    ):
        run.return_value = SimpleNamespace(returncode=1, stdout="")
        assert execute_python(
            "def solve():\n return 42", entrypoint="solve"
        ).sandbox_error
        run.return_value = SimpleNamespace(returncode=0, stdout="not json")
        assert execute_python(
            "def solve():\n return 42", entrypoint="solve"
        ).sandbox_error
        run.return_value = SimpleNamespace(returncode=-9, stdout="")
        result = execute_python("def solve():\n return 42", entrypoint="solve")
        assert result.timed_out and not result.sandbox_error


def test_code_metrics_use_sandbox_return_and_stdout():
    probe = execute_python("def solve():\n return 1", entrypoint="solve")
    if probe.sandbox_error:
        pytest.skip("host does not permit bubblewrap namespaces")
    metrics = TinyGSMMetrics(Tokenizer(), scoring="code", timeout=2)
    example = {
        "input_ids": torch.tensor([0]),
        "prompt": "What is six times seven?",
        "response_ground_truth": "reasoning\n#### 42",
    }
    responses = [
        "```python\ndef simple_math_problem():\n    return 6 * 7\n```",
        'def simple_math_problem():\n    print("answer: 42")',
        "def simple_math_problem():\n    import os\n    return 42",
    ]
    for index, response in enumerate(responses):
        tokens = torch.tensor([[0] + [ord(c) + 3 for c in response] + [1]])
        metrics.update(tokens, [example], [index])
    result = metrics.compute()
    assert result.accuracy == 2 / 3
    assert result.parsed_rate == result.executed_rate == 2 / 3
    assert result.timed_out_rate == result.sandbox_error_rate == 0
    assert [record.correct for record in result.records] == [True, True, False]
    assert all(record.parsed and record.executed for record in result.records[:2])
    assert not result.records[2].parsed and not result.records[2].executed

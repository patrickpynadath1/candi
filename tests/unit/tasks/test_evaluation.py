import json
from types import SimpleNamespace

import datasets
import pytest
import torch

from samplers import SamplingResult

from tasks import evaluation
from tasks.sudoku import SudokuTokenizer, encode_example
from tasks.sudoku_generator import generate_grids


class Logger:
    def info(self, *args, **kwargs):
        pass


def test_sudoku_evaluation_preserves_prompt(monkeypatch, tmp_path):
    tokenizer = SudokuTokenizer()
    puzzle, solution = generate_grids(1, "easy", 11)[0]
    ids, mask = encode_example(puzzle, solution, tokenizer)
    dataset = datasets.DatasetDict(
        {
            "validation": datasets.Dataset.from_dict(
                {"input_ids": [ids], "attention_mask": [mask]}
            ).with_format("torch")
        }
    )

    class Model:
        num_tokens = 180
        device = torch.device("cpu")

        def generate_samples(self, count, **kwargs):
            assert kwargs["prompt_mask"][0, : tokenizer.prompt_len].all()
            assert not kwargs["prompt_mask"][0, tokenizer.prompt_len :].any()
            return SamplingResult(torch.tensor([ids]), {"nfe": 2})

    monkeypatch.setattr(evaluation, "get_sudoku_dataset", lambda *args: dataset)
    monkeypatch.setattr(evaluation, "_prepare_model", lambda *args: Model())
    config = SimpleNamespace(
        sudoku=SimpleNamespace(batch_size=2, output_dir=str(tmp_path)),
        sampling=SimpleNamespace(steps=2),
    )
    result = evaluation.evaluate_sudoku(None, config, tokenizer, None, Logger())
    assert result["accuracy"] == 1.0
    assert json.loads((tmp_path / "results.json").read_text())["num_correct"] == 1


def test_gsm8k_can_score_existing_predictions(tmp_path):
    predictions = [
        {"prompt": "q", "response_ground_truth": "#### 4", "response": "answer: 4"},
        {"prompt": "q", "response_ground_truth": "#### 5", "response": "answer: 2"},
    ]
    input_path = tmp_path / "predictions.json"
    input_path.write_text(json.dumps(predictions))
    output = tmp_path / "output"
    config = SimpleNamespace(
        gsm8k=SimpleNamespace(
            input_file=str(input_path),
            output_dir=str(output),
            scoring="numeric",
            timeout=1.0,
            bootstrap_size=1,
        ),
        seed=1,
    )
    result = evaluation.evaluate_gsm8k(None, config, None, None, Logger())
    assert result["accuracy"] == 0.5
    assert (output / "results.json").exists()


def test_exact_length_batches_preserve_singletons():
    dataset = [
        {"input_ids": [1, 2]},
        {"input_ids": [3, 4, 5]},
        {"input_ids": [6, 7]},
        {"input_ids": [8, 9]},
    ]
    batches = list(evaluation.exact_length_batches(dataset, range(4), batch_size=2))
    assert batches == [[0, 2], [3], [1]]


def test_gsm8k_generation_uses_equal_length_prompt_batches():
    dataset = [
        {"input_ids": [1, 2], "prompt": "a", "response_ground_truth": "#### 1"},
        {
            "input_ids": [1, 2, 3],
            "prompt": "singleton",
            "response_ground_truth": "#### 2",
        },
        {"input_ids": [1, 4], "prompt": "b", "response_ground_truth": "#### 3"},
    ]

    class Tokenizer:
        eos_token_id = 9

        def decode(self, tokens, skip_special_tokens=True):
            del skip_special_tokens
            return " ".join(str(token) for token in tokens)

    class Model:
        num_tokens = 5
        device = torch.device("cpu")

        def generate_samples(self, count, prompt_tokens, prompt_mask, **kwargs):
            del kwargs
            lengths = prompt_mask.sum(1)
            assert torch.equal(lengths, lengths[:1].expand_as(lengths))
            result = prompt_tokens.clone()
            result[torch.arange(count), lengths] = 9
            return result

    records = evaluation.generate_gsm8k_records(
        Model(), dataset, Tokenizer(), batch_size=2, num_steps=1
    )
    assert sorted(record["dataset_index"] for record in records) == [0, 1, 2]
    assert all(record["hit_eos"] for record in records)


def test_gsm8k_bootstrap_is_seeded_and_empty_inputs_are_defined(tmp_path):
    predictions = [
        {
            "prompt": str(index),
            "response_ground_truth": f"#### {index}",
            "response": f"answer: {index if index < 3 else -1}",
        }
        for index in range(5)
    ]
    input_path = tmp_path / "predictions.json"
    input_path.write_text(json.dumps(predictions))

    def run(output_dir, source=input_path):
        config = SimpleNamespace(
            gsm8k=SimpleNamespace(
                input_file=str(source),
                output_dir=str(output_dir),
                scoring="numeric",
                timeout=1.0,
                bootstrap_size=200,
            ),
            seed=17,
        )
        return evaluation.evaluate_gsm8k(None, config, None, None, Logger())

    first = run(tmp_path / "first")
    second = run(tmp_path / "second")
    assert first["accuracy"] == pytest.approx(0.6)
    assert first["accuracy_se"] > 0
    assert first["accuracy_ci_lo"] <= first["accuracy"] <= first["accuracy_ci_hi"]
    assert first["accuracy_se"] == second["accuracy_se"]
    assert first["accuracy_ci_lo"] == second["accuracy_ci_lo"]
    assert first["accuracy_ci_hi"] == second["accuracy_ci_hi"]

    empty_path = tmp_path / "empty.json"
    empty_path.write_text("[]")
    empty = run(tmp_path / "empty-output", empty_path)
    assert empty["accuracy"] == 0.0
    assert empty["num_total"] == 0

import json
from types import SimpleNamespace

import datasets
import torch

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
            return torch.tensor([ids])

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

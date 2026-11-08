"""task=sudoku_diversity: weakened puzzles, repeated generations, diversity counts."""

import json
from types import SimpleNamespace

import pytest
import torch

from core.data.sudoku import SudokuTokenizer, encode_example, sudoku_metrics
from core.sweeps.configuration import compose
from core.tasks.sudoku_diversity import canonical_board, diversity_summary, remove_hints
from core.training.runner import run

GRID = [[(r * 3 + r // 3 + c) % 9 + 1 for c in range(9)] for r in range(9)]


def test_hint_subsets_are_nested_and_answers_untouched():
    tokens, mask = encode_example(GRID, GRID, SudokuTokenizer())
    example = {"input_ids": torch.tensor(tokens), "attention_mask": torch.tensor(mask)}
    previous = set(range(89))
    for fraction in (1.0, 0.75, 0.5, 0.25, 0.0):
        changed, original, retained = remove_hints(example, fraction, 7)
        puzzle = changed["input_ids"][1:90]
        clues = set(((puzzle >= 1) & (puzzle <= 9)).nonzero().flatten().tolist())
        assert clues <= previous
        assert len(clues) == retained and original == 81
        assert torch.equal(changed["input_ids"][91:], example["input_ids"][91:])
        previous = clues
    assert torch.equal(example["input_ids"], torch.tensor(tokens))


def test_counts_alternative_valid_solutions_and_duplicates():
    tokenizer = SudokuTokenizer()
    shifted = [[x % 9 + 1 for x in row] for row in GRID]
    tokens, _ = encode_example([[0] * 9 for _ in range(9)], GRID, tokenizer)
    other, _ = encode_example([[0] * 9 for _ in range(9)], shifted, tokenizer)
    target, alternate = torch.tensor(tokens[91:]), torch.tensor(other[91:])
    # An empty puzzle accepts any valid grid, not only the reference.
    assert sudoku_metrics(alternate, target, tokens[1:90])["valid_solution"]
    rows = [
        SimpleNamespace(index=i, generated=tokenizer.decode(x), valid_solution=True)
        for i, x in enumerate((target, target, alternate))
    ]
    rows.append(SimpleNamespace(index=3, generated="_ " * 89, valid_solution=False))
    result = diversity_summary(rows, [(0, 20, 0)], repetitions=4)
    assert result.valid_count == 3 and result.num_generations == 4
    assert result.unique_count_within_prompts == 2
    assert result.valid_unique_count_within_prompts == 2
    assert result.mean_valid_unique_per_prompt == 2
    assert canonical_board(rows[-1].generated) is None
    with pytest.raises(ValueError, match="multiple of task.repetitions"):
        diversity_summary(rows[:3], [(0, 20, 0)], repetitions=4)


def test_trains_as_sudoku_and_evaluates_diversity(tmp_path):
    settings = compose("smoke", {
        "task": "sudoku_diversity", "task.num_puzzles": 2, "task.repetitions": 3,
        "task.retained_fraction": 0.25, "data.cache_dir": str(tmp_path / "data"),
    })
    settings.output_dir = str(tmp_path / "train")
    settings.trainer.enable_progress_bar = False
    run(settings)
    checkpoint = tmp_path / "train/checkpoints/last.ckpt"
    assert checkpoint.exists()

    settings.mode = "evaluate"
    settings.evaluation.checkpoint = str(checkpoint)
    settings.evaluation.max_examples = None
    settings.output_dir = str(tmp_path / "evaluate")
    run(settings)
    result = json.loads((tmp_path / "evaluate/results.json").read_text())
    assert result["primary_metric"] == {
        "name": "mean_valid_unique_per_prompt", "mode": "max", "dataset": "diversity",
    }
    assert result["num_generations"] == 6 and len(result["records"]) == 6
    assert [p["generations"] for p in result["prompts"]] == [3, 3]
    assert all(p["retained_hints"] < p["original_hints"] for p in result["prompts"])
    assert all("nfe" in record for record in result["records"])
    assert 0 <= result["mean_valid_unique_per_prompt"] <= 3

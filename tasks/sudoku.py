"""Sudoku tokenization, dataset construction, and evaluation metrics."""

from pathlib import Path

import datasets
import torch

from .sudoku_generator import generate_grids


class SudokuTokenizer:
    empty_token_id = 0
    row_separator_id = 10
    bos_token_id = 11
    eos_token_id = 10
    pad_token_id = 0
    mask_token = None
    mask_token_id = None
    vocab_size = 12
    grid_size = 9
    seq_len = 89
    prompt_len = 91
    total_seq_len = 180
    bos_token = "<BOS>"
    eos_token = "|"
    pad_token = "_"

    _symbols = {0: "_", **{i: str(i) for i in range(1, 10)}, 10: "|", 11: "<BOS>"}

    def __len__(self):
        return self.vocab_size

    def decode(self, token_ids, skip_special_tokens=False):
        ignored = {self.bos_token_id} if skip_special_tokens else set()
        return " ".join(
            self._symbols.get(int(token), "?")
            for token in token_ids
            if int(token) not in ignored
        )

    def batch_decode(self, sequences, skip_special_tokens=False):
        return [self.decode(sequence, skip_special_tokens) for sequence in sequences]


def encode_grid(grid, separator=10):
    encoded = []
    for row, values in enumerate(grid):
        encoded.extend(values)
        if row < 8:
            encoded.append(separator)
    return encoded


def encode_example(puzzle, solution, tokenizer):
    input_ids = (
        [tokenizer.bos_token_id]
        + encode_grid(puzzle, tokenizer.row_separator_id)
        + [tokenizer.bos_token_id]
        + encode_grid(solution, tokenizer.row_separator_id)
    )
    attention_mask = [0] * tokenizer.prompt_len + [1] * tokenizer.seq_len
    return input_ids, attention_mask


def get_sudoku_dataset(config, tokenizer):
    if not isinstance(tokenizer, SudokuTokenizer):
        raise TypeError("Sudoku requires SudokuTokenizer")
    data = config.data
    cache = Path(data.cache_dir) / (
        f"sudoku_{data.difficulty}_train{data.num_train}_valid{data.num_valid}"
        f"_seed{data.data_seed}"
    )
    if cache.exists():
        return datasets.load_from_disk(str(cache)).with_format("torch")
    pairs = generate_grids(
        data.num_train + data.num_valid,
        data.difficulty,
        data.data_seed,
        data.sudoku_num_workers,
    )
    records = [
        encode_example(puzzle, solution, tokenizer) for puzzle, solution in pairs
    ]
    split = data.num_train
    dataset = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(
                {
                    "input_ids": [x[0] for x in records[:split]],
                    "attention_mask": [x[1] for x in records[:split]],
                }
            ),
            "validation": datasets.Dataset.from_dict(
                {
                    "input_ids": [x[0] for x in records[split:]],
                    "attention_mask": [x[1] for x in records[split:]],
                }
            ),
        }
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(cache))
    return dataset.with_format("torch")


def solution_grid(tokens):
    values = [
        int(token) for token in tokens if int(token) != SudokuTokenizer.row_separator_id
    ]
    if len(values) != 81:
        return None
    return [values[index : index + 9] for index in range(0, 81, 9)]


def sudoku_metrics(generated, ground_truth, puzzle=None):
    generated = torch.as_tensor(generated)
    ground_truth = torch.as_tensor(ground_truth)
    exact = bool(torch.equal(generated, ground_truth))
    cell_accuracy = float((generated == ground_truth).float().mean())
    grid = solution_grid(generated)
    extras = {}
    if puzzle is not None:
        puzzle = torch.as_tensor(puzzle)
        digits = (ground_truth >= 1) & (ground_truth <= 9)
        clues = (puzzle >= 1) & (puzzle <= 9)
        extras = {
            "digit_accuracy": float((generated[digits] == ground_truth[digits]).float().mean()),
            "clue_consistency": bool((generated[clues] == puzzle[clues]).all()),
        }
    if grid is None:
        return {
            "exact": exact,
            "cell_accuracy": cell_accuracy,
            "valid_rows": 0.0,
            "valid_columns": 0.0,
            "valid_boxes": 0.0,
            "valid_grid": False,
            **extras,
            **({"valid_solution": False} if puzzle is not None else {}),
        }
    target = set(range(1, 10))
    rows = [set(row) == target for row in grid]
    columns = [
        set(grid[row][column] for row in range(9)) == target for column in range(9)
    ]
    boxes = [
        set(grid[row + i][column + j] for i in range(3) for j in range(3)) == target
        for row in (0, 3, 6)
        for column in (0, 3, 6)
    ]
    result = {
        "exact": exact,
        "cell_accuracy": cell_accuracy,
        "valid_rows": sum(rows) / 9,
        "valid_columns": sum(columns) / 9,
        "valid_boxes": sum(boxes) / 9,
        "valid_grid": all(rows + columns + boxes),
        **extras,
    }
    if puzzle is not None:
        format_valid = bool(((generated == 10) == (ground_truth == 10)).all())
        result["valid_solution"] = result["valid_grid"] and extras["clue_consistency"] and format_valid
    return result

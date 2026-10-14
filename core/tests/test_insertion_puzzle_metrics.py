"""Variable-length puzzle answers must not gain validity from output padding."""

import torch
from core.data.nqueens import enumerate_solutions
from core.data.sudoku import SudokuTokenizer, encode_example
from core.tasks.sudoku_metrics import SudokuMetrics
from core.tasks.nqueens_metrics import NQueensMetrics


def test_queens_respects_generated_lengths():
    solution = torch.tensor(enumerate_solutions(8)[0])
    tokens = torch.cat([torch.zeros(8, dtype=torch.long), solution])
    examples = [{"input_ids": tokens}]
    for length, expected in [(16, 1.0), (15, 0.0), (17, 0.0)]:
        sample = torch.cat([tokens, torch.tensor([1])])[None]
        metrics = NQueensMetrics(8, 92)
        metrics.update(sample, examples, [0], lengths=[length])
        assert metrics.compute().valid_solution_rate == expected


def test_sudoku_respects_generated_lengths():
    grid = [
        [(row * 3 + row // 3 + col) % 9 + 1 for col in range(9)] for row in range(9)
    ]
    tokenizer = SudokuTokenizer()
    tokens, _ = encode_example(grid, grid, tokenizer)
    tokens = torch.tensor(tokens)
    examples = [{"input_ids": tokens}]
    for length, expected in [(180, 1.0), (179, 0.0), (181, 0.0)]:
        metrics = SudokuMetrics(tokenizer)
        metrics.update(
            torch.cat([tokens, torch.tensor([1])])[None],
            examples,
            [0],
            lengths=[length],
        )
        assert metrics.compute().valid_solution_rate == expected

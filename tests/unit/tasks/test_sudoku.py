import torch

from tasks.sudoku import SudokuTokenizer, encode_example, solution_grid, sudoku_metrics
from tasks.sudoku_generator import count_solutions, generate_grids


def test_sudoku_generation_is_deterministic_and_unique():
    first = generate_grids(1, "easy", 7)
    second = generate_grids(1, "easy", 7)
    assert first == second
    puzzle, solution = first[0]
    assert sum(value != 0 for row in puzzle for value in row) == 40
    assert count_solutions([row[:] for row in puzzle]) == 1
    assert all(set(row) == set(range(1, 10)) for row in solution)


def test_sudoku_generation_matches_across_worker_counts():
    assert generate_grids(4, "easy", 7, num_workers=2) == generate_grids(
        4, "easy", 7, num_workers=1
    )


def test_sudoku_encoding_has_answer_only_mask():
    tokenizer = SudokuTokenizer()
    puzzle, solution = generate_grids(1, "easy", 3)[0]
    ids, mask = encode_example(puzzle, solution, tokenizer)
    assert len(ids) == tokenizer.total_seq_len
    assert len(mask) == tokenizer.total_seq_len
    assert mask[: tokenizer.prompt_len] == [0] * tokenizer.prompt_len
    assert mask[tokenizer.prompt_len :] == [1] * tokenizer.seq_len
    assert solution_grid(ids[tokenizer.prompt_len :]) == solution


def test_sudoku_metrics_distinguish_exact_and_valid():
    _, solution = generate_grids(1, "easy", 5)[0]
    tokenizer = SudokuTokenizer()
    encoded = torch.tensor(
        encode_example(solution, solution, tokenizer)[0][tokenizer.prompt_len :]
    )
    metrics = sudoku_metrics(encoded, encoded)
    assert metrics["exact"] and metrics["valid_grid"]
    corrupted = encoded.clone()
    corrupted[0] = corrupted[1]
    metrics = sudoku_metrics(corrupted, encoded)
    assert not metrics["exact"] and not metrics["valid_grid"]


def test_sudoku_metrics_reject_malformed_and_out_of_range_grids():
    _, solution = generate_grids(1, "easy", 11)[0]
    tokenizer = SudokuTokenizer()
    encoded = torch.tensor(
        encode_example(solution, solution, tokenizer)[0][tokenizer.prompt_len :]
    )

    malformed = sudoku_metrics(encoded[:-1], encoded[:-1])
    assert malformed["exact"]
    assert not malformed["valid_grid"]
    assert malformed["valid_rows"] == 0.0

    invalid_token = encoded.clone()
    invalid_token[0] = 0
    invalid = sudoku_metrics(invalid_token, encoded)
    assert not invalid["exact"]
    assert not invalid["valid_grid"]


def test_sudoku_tokenizer_contract():
    tokenizer = SudokuTokenizer()
    assert len(tokenizer) == 12
    assert tokenizer.prompt_len == 91
    assert "<BOS>" in tokenizer.decode([11, 1, 10])


def test_valid_grid_must_obey_clues_and_digit_accuracy_excludes_separators():
    puzzle, solution = generate_grids(1, "easy", 5)[0]
    encoded, _ = encode_example(puzzle, solution, SudokuTokenizer())
    truth, clue_tokens = torch.tensor(encoded[91:]), torch.tensor(encoded[1:90])
    result = sudoku_metrics(truth, truth, clue_tokens)
    assert result["valid_solution"] and result["digit_accuracy"] == 1
    # A digit permutation remains a valid grid but violates the puzzle clues.
    other = torch.where(truth == 1, 2, torch.where(truth == 2, 1, truth))
    result = sudoku_metrics(other, truth, clue_tokens)
    assert result["valid_grid"] and not result["valid_solution"]
    assert not result["clue_consistency"]
    wrong = truth.clone()
    wrong[0] = 0
    result = sudoku_metrics(wrong, truth, clue_tokens)
    assert abs(result["digit_accuracy"] - 80/81) < 1e-6

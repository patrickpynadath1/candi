"""Deterministic generation of unique-solution Sudoku tasks."""

import random
from multiprocessing import Pool

DIFFICULTY_TO_CLUES = {"easy": 40, "medium": 35, "hard": 30}


def is_valid(board, row, column, number):
    if number in board[row] or any(
        board[index][column] == number for index in range(9)
    ):
        return False
    box_row, box_column = row - row % 3, column - column % 3
    return all(
        board[box_row + i][box_column + j] != number for i in range(3) for j in range(3)
    )


def _fill_grid(grid, rng):
    for row in range(9):
        for column in range(9):
            if grid[row][column] == 0:
                numbers = list(range(1, 10))
                rng.shuffle(numbers)
                for number in numbers:
                    if is_valid(grid, row, column, number):
                        grid[row][column] = number
                        if _fill_grid(grid, rng):
                            return True
                        grid[row][column] = 0
                return False
    return True


def count_solutions(grid, limit=2):
    count = 0

    def solve():
        nonlocal count
        if count >= limit:
            return
        for row in range(9):
            for column in range(9):
                if grid[row][column] == 0:
                    for number in range(1, 10):
                        if is_valid(grid, row, column, number):
                            grid[row][column] = number
                            solve()
                            grid[row][column] = 0
                    return
        count += 1

    solve()
    return count


def _generate_one(arguments):
    seed, clues = arguments
    rng = random.Random(seed)
    solution = [[0] * 9 for _ in range(9)]
    _fill_grid(solution, rng)
    puzzle = [row[:] for row in solution]
    cells = [(row, column) for row in range(9) for column in range(9)]
    rng.shuffle(cells)
    removed = 0
    for row, column in cells:
        if removed == 81 - clues:
            break
        value = puzzle[row][column]
        puzzle[row][column] = 0
        if count_solutions([item[:] for item in puzzle]) == 1:
            removed += 1
        else:
            puzzle[row][column] = value
    return puzzle, solution


def generate_grids(count, difficulty, seed, num_workers=1):
    if difficulty not in DIFFICULTY_TO_CLUES:
        raise ValueError(f"unknown Sudoku difficulty: {difficulty}")
    clues = DIFFICULTY_TO_CLUES[difficulty]
    results, seen, next_seed = [], set(), seed
    while len(results) < count:
        batch_count = count - len(results)
        jobs = [(next_seed + i, clues) for i in range(batch_count)]
        next_seed += batch_count
        if num_workers > 1:
            with Pool(num_workers) as pool:
                candidates = pool.map(_generate_one, jobs)
        else:
            candidates = map(_generate_one, jobs)
        for puzzle, solution in candidates:
            key = tuple(value for row in solution for value in row)
            if key not in seen:
                seen.add(key)
                results.append((puzzle, solution))
    random.Random(seed).shuffle(results)
    return results

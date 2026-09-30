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
    """Count up to limit using bit masks and minimum-remaining-values search.

    Does not modify the caller's grid. The search order changes only the cost of
    uniqueness verification, not puzzle generation or its random-number stream.
    """
    count = 0
    rows, columns, boxes = [0] * 9, [0] * 9, [0] * 9
    empty = []
    for row in range(9):
        for column in range(9):
            value = grid[row][column]
            box = row // 3 * 3 + column // 3
            if value == 0:
                empty.append((row, column, box))
            else:
                bit = 1 << (value - 1)
                if (rows[row] | columns[column] | boxes[box]) & bit:
                    return 0
                rows[row] |= bit
                columns[column] |= bit
                boxes[box] |= bit

    def solve(depth=0):
        nonlocal count
        if count >= limit:
            return
        if depth == len(empty):
            count += 1
            return
        best, candidates, size = depth, 0, 10
        for index in range(depth, len(empty)):
            row, column, box = empty[index]
            choices = 511 & ~(rows[row] | columns[column] | boxes[box])
            n = choices.bit_count()
            if n < size:
                best, candidates, size = index, choices, n
                if n <= 1:
                    break
        if not candidates:
            return
        empty[depth], empty[best] = empty[best], empty[depth]
        row, column, box = empty[depth]
        while candidates and count < limit:
            bit = candidates & -candidates
            candidates ^= bit
            rows[row] |= bit
            columns[column] |= bit
            boxes[box] |= bit
            solve(depth + 1)
            rows[row] ^= bit
            columns[column] ^= bit
            boxes[box] ^= bit
        empty[depth], empty[best] = empty[best], empty[depth]

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


def generate_grids(count, difficulty, seed, num_workers=1, exact_clues=False):
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
            if (
                exact_clues
                and sum(value != 0 for row in puzzle for value in row) != clues
            ):
                continue
            key = tuple(value for row in solution for value in row)
            if key not in seen:
                seen.add(key)
                results.append((puzzle, solution))
    random.Random(seed).shuffle(results)
    return results

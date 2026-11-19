"""How copy-like is each rung of the Sudoku hint ladder?

For the same 32 validation puzzles and nested hint subsets as task=sudoku_diversity
(remove_hints, seed + puzzle index), report per retained fraction:
  hints      mean number of hints kept
  peeled     share of blank cells the peeling decoder fills from the hints alone
             (repeat: a cell with one candidate left, or a digit with one place left in a
             row, column or box, is filled; this is erasure-channel belief propagation)
  solved     share of puzzles peeling solves completely
  sols>=16   share of puzzles with at least 16 solutions (so 16 samples could all differ)

    uv run --no-project --with torch --with datasets python sudoku_peeling.py [--selfcheck]
"""

import json
import sys
from pathlib import Path

import torch

CACHE = Path(__file__).resolve().parents[2] / "core/data_cache/sudoku_hard_train48000_valid2000_seed42"
FRACTIONS = [1.0, 0.75, 0.5, 0.25, 0.1, 0.0]
NUM_PUZZLES, SEED, PROMPT_LEN = 32, 0, 91
UNITS = ([[r * 9 + c for c in range(9)] for r in range(9)] + [[r * 9 + c for r in range(9)] for c in range(9)]
         + [[(br + r) * 9 + bc + c for r in range(3) for c in range(3)] for br in (0, 3, 6) for bc in (0, 3, 6)])
PEERS = [sorted({p for u in UNITS if i in u for p in u} - {i}) for i in range(81)]
ALL = 0x3FE  # bits 1..9


def weakened_grid(input_ids, fraction, seed):
    """Exactly candi's remove_hints: keep a seeded fraction of the clues, as an 81-cell grid."""
    puzzle = input_ids[1:PROMPT_LEN - 1].clone()
    positions = ((puzzle >= 1) & (puzzle <= 9)).nonzero().flatten()
    order = torch.randperm(len(positions), generator=torch.Generator().manual_seed(seed))
    keep = int(len(positions) * fraction + 0.5)
    puzzle[positions[order[keep:]]] = 0
    return [int(v) for v in puzzle if v != 10]


def candidates(grid):
    cand = [0 if v else ALL for v in grid]
    for i, v in enumerate(grid):
        if v:
            for p in PEERS[i]:
                cand[p] &= ~(1 << v)
    return cand


def peel(grid):
    """Fill singles until none are left. Returns the new grid (None on a contradiction)."""
    grid = list(grid)
    while True:
        cand, changed = candidates(grid), False
        for i in range(81):
            if not grid[i]:
                if cand[i] == 0:
                    return None
                if cand[i] & (cand[i] - 1) == 0:
                    grid[i], changed = cand[i].bit_length() - 1, True
        for unit in UNITS:
            for d in range(1, 10):
                spots = [i for i in unit if not grid[i] and cand[i] >> d & 1]
                if len(spots) == 1 and d not in (grid[i] for i in unit):
                    grid[spots[0]], changed = d, True
        if not changed:
            return grid
        cand = candidates(grid)


def count_solutions(grid, cap):
    grid = peel(grid)
    if grid is None:
        return 0
    if all(grid):
        return 1
    cand = candidates(grid)
    i = min((i for i in range(81) if not grid[i]), key=lambda i: bin(cand[i]).count("1"))
    total = 0
    for d in range(1, 10):
        if cand[i] >> d & 1:
            grid[i] = d
            total += count_solutions(grid, cap - total)
            grid[i] = 0
            if total >= cap:
                break
    return total


def ladder():
    import datasets
    valid = datasets.load_from_disk(str(CACHE / "validation")).with_format("torch")
    out = []
    for f in FRACTIONS:
        hints = peeled = solved = many = 0
        for k in range(NUM_PUZZLES):
            g = weakened_grid(valid[k]["input_ids"], f, SEED + k)
            blanks = sum(v == 0 for v in g)
            p = peel(g)
            filled = sum(1 for a, b in zip(g, p) if a == 0 and b)
            hints += 81 - blanks
            peeled += filled / blanks
            solved += all(p)
            many += count_solutions(g, 16) >= 16
        n = NUM_PUZZLES
        out.append(dict(fraction=f, hints=hints / n, peeled=round(peeled / n, 3),
                        solved=round(solved / n, 3), sols_ge_16=round(many / n, 3)))
    return out


def selfcheck():
    solved = [int(c) for c in
              "534678912672195348198342567859761423426853791713924856961537284287419635345286179"]
    g = list(solved)
    for i in (0, 10, 40, 80):
        g[i] = 0
    assert peel(g) == solved and count_solutions(g, 5) == 1
    assert count_solutions([0] * 81, 16) == 16
    print("selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        print(json.dumps(ladder(), indent=1))

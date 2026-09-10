"""Shah symbolic clue parser and assignment checker (zero-based source values)."""

import numpy as np

RELATIONS = ("=", "!=", "immediate-left", "nbr", "ends", "left-of", "inbetween")
SYMBOLS = (
    "<PAD>",
    "<BOS>",
    "<ANSWER>",
    "<HOUSES>",
    "<ATTRS>",
    *map(str, range(7)),
    *RELATIONS,
    "LHS",
    "RHS",
    "c",
    "n",
    "CLUE_END",
)


def parse_clues(tokens):
    tokens = list(map(str, tokens))
    if "ANSWER" in tokens:
        tokens = tokens[: tokens.index("ANSWER")]
    clues = []
    chunk = []
    for token in tokens:
        if token != "CLUE_END":
            chunk.append(token)
            continue
        if (
            len(chunk) < 6
            or chunk[0] not in RELATIONS
            or chunk[1] != "LHS"
            or chunk[5] != "RHS"
        ):
            raise ValueError(f"malformed Zebra clue: {chunk}")
        args = chunk[2:5] + chunk[6:]
        required = 1 if chunk[0] == "ends" else 3 if chunk[0] == "inbetween" else 2
        if len(args) != required * 3:
            raise ValueError(f"wrong arity: {chunk}")
        refs = []
        for i in range(0, len(args), 3):
            kind, attr, value = args[i : i + 3]
            if kind not in ("c", "n"):
                raise ValueError(f"unknown reference: {kind}")
            refs.append((kind, int(attr), int(value)))
        clues.append((chunk[0], refs))
        chunk = []
    if chunk:
        raise ValueError("unterminated Zebra clue")
    return clues, tokens


def clue_holds(clue, assignment):
    table = np.asarray(assignment)
    houses = table.shape[1]
    positions = []
    for kind, attr, value in clue[1]:
        if not 0 <= value < houses or (kind == "c" and not 0 <= attr < len(table)):
            return False
        if kind == "n":
            positions.append(value)
        else:
            found = np.flatnonzero(table[attr] == value)
            if len(found) != 1:
                return False
            positions.append(int(found[0]))
    relation = clue[0]
    a = positions[0]
    if relation == "ends":
        return a in (0, houses - 1)
    b = positions[1]
    if relation == "=":
        return a == b
    if relation == "!=":
        return a != b
    if relation == "immediate-left":
        return a + 1 == b
    if relation == "nbr":
        return abs(a - b) == 1
    if relation == "left-of":
        return a < b
    c = positions[2]
    # Released triples are serialized in left-to-right order.
    return a < b < c


def zebra_metrics(prediction, target, clues):
    truth = np.asarray(target)
    pred = np.asarray(prediction).reshape(truth.shape)
    legal = all(set(row) == set(range(truth.shape[1])) for row in pred)
    held = [clue_holds(c, pred) for c in clues]
    return dict(
        exact=bool(np.array_equal(pred, truth)),
        cell_accuracy=float((pred == truth).mean()),
        valid_assignment=legal,
        constraints_satisfied=bool(legal and all(held)),
        clue_accuracy=sum(held) / max(1, len(held)),
        malformed=not legal,
    )


def solve_clues(clues, attributes, houses, limit=2, max_nodes=100000):
    """Small audit solver: enumerate attribute permutations with clue pruning.

    Returns up to ``limit`` assignment tables. Not used during generation.
    A node limit raises instead of incorrectly claiming uniqueness.
    """
    from itertools import permutations

    table = np.full((attributes, houses), -1, dtype=int)
    domains = list(permutations(range(houses)))
    solutions = []
    nodes = 0

    def compatible(assigned):
        return all(
            clue_holds(clue, table)
            for clue in clues
            if all(kind == "n" or attr in assigned for kind, attr, _ in clue[1])
        )

    def search(assigned):
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            raise RuntimeError("Zebra audit solver node limit reached")
        if len(assigned) == attributes:
            solutions.append(table.copy().tolist())
            return
        # Prefer attributes that make the most constraints decidable next.
        remaining = set(range(attributes)) - assigned
        attribute = max(
            remaining,
            key=lambda a: sum(
                all(kind == "n" or attr in assigned | {a} for kind, attr, _ in clue[1])
                for clue in clues
            ),
        )
        for row in domains:
            table[attribute] = row
            if compatible(assigned | {attribute}):
                search(assigned | {attribute})
                if len(solutions) >= limit:
                    return
        table[attribute] = -1

    search(set())
    return solutions

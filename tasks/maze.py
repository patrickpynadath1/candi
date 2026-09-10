"""Independent validation of EqR Maze-Unique board/path labels."""

from collections import deque
import numpy as np

# Released EqR IDs: PAD=0, wall=1, space=2, start=3, goal=4, path=5.


def shortest_path(board):
    board = np.asarray(board)
    starts, goals = np.argwhere(board == 3), np.argwhere(board == 4)
    if len(starts) != 1 or len(goals) != 1:
        raise ValueError("maze must contain exactly one start and goal")
    start, goal = tuple(starts[0]), tuple(goals[0])
    queue = deque([start])
    dist = {start: 0}
    counts = {start: 1}
    parent = {}
    while queue:
        p = queue.popleft()
        for q in neighbors(p, board.shape):
            if board[q] == 1:
                continue
            if q not in dist:
                dist[q] = dist[p] + 1
                counts[q] = counts[p]
                parent[q] = p
                queue.append(q)
            elif dist[q] == dist[p] + 1:
                counts[q] = min(2, counts[q] + counts[p])
    if goal not in dist:
        raise ValueError("maze has no start-to-goal path")
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return path[::-1], counts[goal] == 1


def neighbors(p, shape):
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        q = p[0] + dy, p[1] + dx
        if 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]:
            yield q


def maze_metrics(prediction, target, puzzle):
    board = np.asarray(puzzle)
    pred = np.asarray(prediction).reshape(board.shape)
    truth = np.asarray(target).reshape(board.shape)
    reference, unique = shortest_path(board)
    selected = set(map(tuple, np.argwhere(np.isin(pred, [3, 4, 5]))))
    start, goal = reference[0], reference[-1]
    preserved = (
        np.all(pred[board == 1] == 1)
        and np.all(pred[board == 3] == 3)
        and np.all(pred[board == 4] == 4)
    )
    legal = np.all(np.isin(pred[board == 2], [2, 5]))
    connected = set()
    queue = [start] if start in selected else []
    while queue:
        p = queue.pop()
        if p in connected:
            continue
        connected.add(p)
        queue.extend(
            q for q in neighbors(p, board.shape) if q in selected and q not in connected
        )
    degrees = all(
        sum(q in selected for q in neighbors(p, board.shape))
        == (1 if p in (start, goal) else 2)
        for p in selected
    )
    valid = bool(
        preserved and legal and goal in connected and connected == selected and degrees
    )
    return dict(
        exact=bool(np.array_equal(pred, truth)),
        cell_accuracy=float((pred == truth).mean()),
        valid_path=valid,
        optimal_path=valid and len(selected) == len(reference),
        length_error=abs(len(selected) - len(reference)),
        unique_shortest_path=unique,
    )

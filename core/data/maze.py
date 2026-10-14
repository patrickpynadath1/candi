"""Versioned maze-dataset maps with independent BFS labels and cached splits."""

from collections import deque
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import tempfile

import numpy as np

GENERATOR_VERSION = "1.4.2"
FORMAT_VERSION = 1


def neighbors(cells, node, size):
    row, col = divmod(node, size)
    for bit, dr, dc in ((1, -1, 0), (2, 0, 1), (4, 1, 0), (8, 0, -1)):
        r, c = row + dr, col + dc
        if cells[node] & bit and 0 <= r < size and 0 <= c < size:
            yield r * size + c


def shortest_path(cells, start, goal, size):
    """BFS independent of the generator and learned model."""
    parents = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == goal:
            result = [node]
            while parents[node] is not None:
                node = parents[node]
                result.append(node)
            return result[::-1]
        for nxt in neighbors(cells, node, size):
            if nxt not in parents:
                parents[nxt] = node
                queue.append(nxt)
    raise ValueError("disconnected maze endpoints")


def route_through(cells, anchors, size):
    route = [anchors[0]]
    for start, goal in zip(anchors, anchors[1:]):
        route.extend(shortest_path(cells, start, goal, size)[1:])
    return route


def cell_connections(connection_list):
    size = connection_list.shape[1]
    cells = [0] * (size * size)
    for r in range(size):
        for c in range(size):
            node = r * size + c
            if r + 1 < size and connection_list[0, r, c]:
                cells[node] |= 4
                cells[node + size] |= 1
            if c + 1 < size and connection_list[1, r, c]:
                cells[node] |= 2
                cells[node + 1] |= 8
    return cells


def map_identity(cells, size):
    """Canonical topology under rotations/reflections, excluding endpoint choices."""
    edges = [
        (u, v) for u in range(size * size) for v in neighbors(cells, u, size) if u < v
    ]
    variants = []
    for reflect in (False, True):
        for turns in range(4):

            def transform(node):
                r, c = divmod(node, size)
                if reflect:
                    c = size - 1 - c
                for _ in range(turns):
                    r, c = c, size - 1 - r
                return r * size + c

            transformed = sorted(
                tuple(sorted((transform(u), transform(v)))) for u, v in edges
            )
            variants.append(json.dumps(transformed, separators=(",", ":")))
    return hashlib.sha256(min(variants).encode()).hexdigest()


def generate_splits(config):
    size = int(config["grid_size"])
    count = int(config["num_waypoints"])
    families = list(config["families"])
    counts = [int(config[k]) for k in ("num_train", "num_valid", "num_test")]
    if size < 2 or not 0 <= count <= size * size - 2 or any(n < 1 for n in counts):
        raise ValueError("invalid grid size, waypoint count or split sizes")
    if (
        not families
        or len(set(families)) != len(families)
        or set(families) - {"tree", "cyclic"}
    ):
        raise ValueError("families must be distinct tree/cyclic choices")
    if not 1 <= config["min_path_edges"] <= config["max_path_edges"]:
        raise ValueError("invalid path length range")
    if not 0 < config["cycle_probability"] < 1:
        raise ValueError("cycle_probability must be in (0, 1)")
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        from maze_dataset.generation import LatticeMazeGenerators

        if version("maze-dataset") != GENERATOR_VERSION:
            raise ValueError(
                f"install maze-dataset=={GENERATOR_VERSION} for reproducible generation"
            )
        random.seed(config["data_seed"])
        np.random.seed(config["data_seed"])
        rng = random.Random(config["data_seed"])
        seen, splits = set(), {}
        for split, n in zip(("train", "validation", "test"), counts):
            examples = []
            for _ in range(max(1000, n * 100)):
                if len(examples) == n:
                    break
                family = families[len(examples) % len(families)]
                maze = (
                    LatticeMazeGenerators.gen_dfs((size, size))
                    if family == "tree"
                    else LatticeMazeGenerators.gen_dfs_percolation(
                        (size, size), p=config["cycle_probability"]
                    )
                )
                cells = cell_connections(maze.connection_list)
                if (
                    family == "cyclic"
                    and sum(v.bit_count() for v in cells) // 2 < size * size
                ):
                    continue
                identity = map_identity(cells, size)
                if identity in seen:
                    continue
                anchors = rng.sample(range(size * size), count + 2)
                path = route_through(cells, anchors, size)
                if (
                    not config["min_path_edges"]
                    <= len(path) - 1
                    <= config["max_path_edges"]
                ):
                    continue
                seen.add(identity)
                examples.append(
                    dict(
                        maze_id=identity,
                        family=family,
                        grid_size=size,
                        cells=cells,
                        anchors=anchors,
                        path=path,
                    )
                )
            if len(examples) != n:
                raise ValueError(
                    "could not generate enough distinct maps; reduce split counts or relax path filters"
                )
            splits[split] = examples
        return splits
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def load_maze_dataset(config):
    import fcntl

    options = {
        k: config[k]
        for k in (
            "grid_size",
            "num_waypoints",
            "families",
            "num_train",
            "num_valid",
            "num_test",
            "data_seed",
            "min_path_edges",
            "max_path_edges",
            "cycle_probability",
        )
    }
    options["families"] = list(options["families"])
    manifest = dict(format=FORMAT_VERSION, generator=GENERATOR_VERSION, options=options)
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:20]
    root = Path(config["cache_dir"]) / f"maze_{key}"
    root.mkdir(parents=True, exist_ok=True)
    target = root / "dataset.json"
    with (root / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            payload = dict(manifest=manifest, splits=generate_splits(options))
            with tempfile.NamedTemporaryFile(
                mode="w", dir=root, delete=False
            ) as handle:
                json.dump(payload, handle)
                temporary = Path(handle.name)
            temporary.replace(target)
        payload = json.loads(target.read_text())
        if payload["manifest"] != manifest:
            raise ValueError("maze cache manifest mismatch")
    return payload["splits"]

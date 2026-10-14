"""Exact graph checks accept any valid route, independent of the label path."""

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from core.data.maze import neighbors, route_through


@dataclass(frozen=True)
class MazeRecord:
    index: int
    maze_id: str
    family: str
    path: list[int]
    hit_eos: bool
    prompt_preserved: bool
    valid_route: bool
    waypoint_success: bool
    success: bool
    optimal: bool
    path_edges: int
    shortest_edges: int
    excess_edges: int | None


@dataclass(frozen=True)
class MazeAggregate:
    success_rate: float
    valid_route_rate: float
    waypoint_success_rate: float
    optimal_solution_rate: float
    eos_rate: float
    prompt_preservation_rate: float
    mean_path_edges: float
    mean_excess_edges_successful: float | None
    mean_unique_valid_routes_per_maze: float
    num_total: int
    by_family: dict
    records: list[MazeRecord]


class MazeMetrics:
    supports_variable_length = True

    def __init__(self, task):
        self.task, self.records = task, []

    def update(self, samples, examples, indices, *, lengths=None):
        t = self.task.tokenizer
        for row, (sample, example, index) in enumerate(zip(samples, examples, indices)):
            sequence = (
                sample[: lengths[row]].tolist()
                if lengths is not None
                else sample.tolist()
            )
            prefix = self.task.prompt(example)
            preserved = sequence[: len(prefix)] == prefix
            completion = sequence[len(prefix) :]
            eos = t.eos_token_id in completion
            if eos:
                completion = completion[: completion.index(t.eos_token_id)]
            path = [v - t.node_offset for v in completion]
            size = example["grid_size"]
            anchors = example["anchors"]
            in_bounds = bool(path) and all(0 <= v < size * size for v in path)
            legal = in_bounds and all(
                v in neighbors(example["cells"], u, size)
                for u, v in zip(path, path[1:])
            )
            route = bool(legal and path[0] == anchors[0] and path[-1] == anchors[-1])
            reached = 0
            for node in path:
                if reached < len(anchors) and node == anchors[reached]:
                    reached += 1
            waypoint = bool(route and reached == len(anchors))
            success = preserved and eos and waypoint
            shortest = len(route_through(example["cells"], anchors, size)) - 1
            edges = max(0, len(path) - 1)
            self.records.append(
                MazeRecord(
                    index,
                    example["maze_id"],
                    example["family"],
                    path,
                    eos,
                    preserved,
                    route,
                    waypoint,
                    success,
                    success and edges == shortest,
                    edges,
                    shortest,
                    edges - shortest if success else None,
                )
            )

    def compute(self):
        if not self.records:
            raise ValueError("cannot score empty maze samples")
        records = self.records
        routes, families = defaultdict(set), defaultdict(list)
        for r in records:
            routes[r.maze_id]
            if r.success:
                routes[r.maze_id].add(tuple(r.path))
            families[r.family].append(r)
        excess = [r.excess_edges for r in records if r.success]
        return MazeAggregate(
            mean(r.success for r in records),
            mean(r.valid_route for r in records),
            mean(r.waypoint_success for r in records),
            mean(r.optimal for r in records),
            mean(r.hit_eos for r in records),
            mean(r.prompt_preserved for r in records),
            mean(r.path_edges for r in records),
            mean(excess) if excess else None,
            mean(len(paths) for paths in routes.values()),
            len(records),
            {
                family: dict(
                    success_rate=mean(r.success for r in rows),
                    optimal_solution_rate=mean(r.optimal for r in rows),
                    num_total=len(rows),
                )
                for family, rows in families.items()
            },
            list(records),
        )

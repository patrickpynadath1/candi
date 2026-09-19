"""Exact ordered-waypoint distribution under one global horizon.

The state is (next required waypoint, current node). Visiting a future waypoint
before its turn is forbidden; revisiting already satisfied waypoints is allowed.
"""

import numpy as np
from scipy.special import logsumexp
from .graph import N_SPECIAL, PAD


class RouteDistribution:
    def __init__(self, rsp, waypoints, hints=None, *, _defer=False):
        self.rsp, self.graph, self.config = rsp, rsp.graph, rsp.config
        self.waypoints = tuple(int(v) for v in waypoints)
        self.hints = hints or {}
        if len(self.waypoints) < 2 or len(set(self.waypoints)) != len(self.waypoints):
            raise ValueError(
                "waypoints must contain at least two distinct node ids, with no duplicates"
            )
        if any(v < 0 or v >= self.graph.n_nodes for v in self.waypoints):
            raise ValueError("waypoint outside graph")
        if any(
            p < 0 or p > self.config.max_steps or v < 0 or v >= self.graph.n_nodes
            for p, v in self.hints.items()
        ):
            raise ValueError("hint outside path/node range")
        self.stages = len(self.waypoints) - 1
        self.next_stage = []
        self.allowed = []
        for q in range(self.stages):
            self.next_stage.append(
                np.where(
                    rsp.cols == self.waypoints[q + 1], min(q + 1, self.stages - 1), q
                )
            )
            self.allowed.append(~np.isin(rsp.cols, self.waypoints[q + 2 :]))
        if _defer:
            return
        key = "_".join(map(str, self.waypoints[1:]))
        cache = rsp.cache_dir
        zpath = cache / f"suffix_{key}.npy" if cache and not hints else None
        hpath = cache / f"suffix_{key}_entropy.npy" if zpath else None
        if zpath and zpath.exists() and hpath.exists():
            self.z = np.load(zpath, mmap_mode="r")
            self.entropies = np.load(hpath, mmap_mode="r")
            self.entropy = float(self.entropies[self.waypoints[0]])
        else:
            self.z, self.entropy = self._tables()

    @classmethod
    def batch(cls, rsp, waypoint_lists):
        """Compute several suffix tables in dense columns of each sparse update."""
        routes = [cls(rsp, points, _defer=True) for points in waypoint_lists]
        if not routes:
            return []
        stages = routes[0].stages
        if any(route.stages != stages for route in routes):
            raise ValueError("batched routes must have the same waypoint count")
        r, c, n, batch = rsp, rsp.config, rsp.graph.n_nodes, len(routes)
        columns = np.arange(batch)
        z = np.full((c.max_steps + 1, stages, n, batch), -np.inf)
        h = np.zeros((stages, n, batch))
        next_stages = np.stack([route.next_stage for route in routes], axis=-1)
        allowed = np.stack([route.allowed for route in routes], axis=-1)
        for p in range(c.max_steps, -1, -1):
            updated = np.zeros_like(h)
            for q in range(stages):
                if p < c.max_steps:
                    values = (
                        r.log_w[:, None]
                        + z[p + 1, next_stages[q], r.cols[:, None], columns]
                    )
                    values = np.where(allowed[q], values, -np.inf)
                    z[p, q] = r._row_logsumexp(values)
                    safe = np.where(np.isfinite(z[p, q]), z[p, q], 0.0)
                    lp = values - safe[r.rows]
                    contribution = np.exp(lp) * (
                        h[next_stages[q], r.cols[:, None], columns]
                        - np.where(np.isfinite(lp), lp, 0.0)
                    )
                    updated[q] = r._row_sum(contribution)
                for b, route in enumerate(routes):
                    if q == stages - 1:
                        z[p, q, route.waypoints[-1], b] = (
                            0.0
                            if c.length_mode == "at_most" or p == c.max_steps
                            else -np.inf
                        )
                        updated[q, route.waypoints[-1], b] = 0.0
                    else:
                        forbidden = list(route.waypoints[q + 1 :])
                        z[p, q, forbidden, b] = -np.inf
                        updated[q, forbidden, b] = 0.0
            h = updated
        for b, route in enumerate(routes):
            route.z = z[:, :, :, b]
            route.entropies = h[0, :, b] / np.log(2)
            route.entropy = float(route.entropies[route.waypoints[0]])
        return routes

    def _tables(self):
        r, c, n = self.rsp, self.config, self.graph.n_nodes
        z = np.full((c.max_steps + 1, self.stages, n), -np.inf)
        h = np.zeros((self.stages, n))
        for p in range(c.max_steps, -1, -1):
            next_h = np.zeros_like(h)
            for q in range(self.stages):
                if p < c.max_steps:
                    values = r.log_w + z[p + 1, self.next_stage[q], r.cols]
                    values = np.where(self.allowed[q], values, -np.inf)
                    z[p, q] = r._row_logsumexp(values[:, None])[:, 0]
                    safe = np.where(np.isfinite(z[p, q]), z[p, q], 0.0)
                    lp = values - safe[r.rows]
                    contribution = np.exp(lp) * (
                        h[self.next_stage[q], r.cols]
                        - np.where(np.isfinite(lp), lp, 0.0)
                    )
                    next_h[q] = r._row_sum(contribution)
                if q == self.stages - 1:
                    allow_stop = (
                        c.length_mode == "at_most" or p == c.max_steps
                    ) and not any(i > p for i in self.hints)
                    z[p, q, self.waypoints[-1]] = 0.0 if allow_stop else -np.inf
                    next_h[q, self.waypoints[-1]] = 0.0
                # Unreachable stage states cannot be used to bypass waypoint order.
                forbidden = list(self.waypoints[q + 1 :]) if q < self.stages - 1 else []
                z[p, q, forbidden] = -np.inf
                next_h[q, forbidden] = 0.0
                if p in self.hints:
                    excluded = np.arange(n) != self.hints[p]
                    z[p, q, excluded] = -np.inf
                    next_h[q, excluded] = 0.0
            h = next_h
        self.entropies = h[0] / np.log(2)
        return z, float(self.entropies[self.waypoints[0]])

    def with_start(self, start):
        import copy

        if start in self.waypoints[1:]:
            raise ValueError("start duplicates a waypoint")
        result = copy.copy(self)
        result.waypoints = (int(start),) + self.waypoints[1:]
        result.entropy = float(self.entropies[start])
        return result

    @property
    def log_partition(self):
        return float(self.z[0, 0, self.waypoints[0]])

    def sample(self, rng):
        if not np.isfinite(self.log_partition):
            raise ValueError("waypoints/hints have zero probability within max_steps")
        r = self.rsp
        for _ in range(self.config.max_sampling_attempts):
            path, q = [self.waypoints[0]], 0
            for p in range(self.config.max_steps):
                i = path[-1]
                if i == self.waypoints[-1]:
                    break
                a, b = r.ptr[i : i + 2]
                scores = (
                    r.log_w[a:b] + self.z[p + 1, self.next_stage[q][a:b], r.cols[a:b]]
                )
                scores = np.where(self.allowed[q][a:b], scores, -np.inf)
                edge = int(
                    rng.choice(np.arange(a, b), p=np.exp(scores - logsumexp(scores)))
                )
                path.append(int(r.cols[edge]))
                q = int(self.next_stage[q][edge])
            if not self.config.require_simple or len(path) == len(set(path)):
                return path
        raise RuntimeError("simple-path rejection exhausted max_sampling_attempts")

    def true_logprob(self, path):
        from .verify import verify_route

        valid = verify_route(
            self.graph,
            self.waypoints,
            path,
            max_steps=self.config.max_steps,
            require_simple=self.config.require_simple,
            hints=self.hints,
        )
        if (
            not valid.valid
            or (
                self.config.length_mode == "exact"
                and valid.n_steps != self.config.max_steps
            )
            or not np.isfinite(self.log_partition)
        ):
            return -np.inf
        nodes = np.asarray([int(t) - N_SPECIAL for t in path if t != PAD])
        keys = nodes[:-1] * self.graph.n_nodes + nodes[1:]
        return float(
            self.rsp.log_w[np.searchsorted(self.graph.edge_keys, keys)].sum()
            - self.log_partition
        )

    def cond_entropy_bits(self):
        return self.entropy if np.isfinite(self.log_partition) else float("nan")

    def first_step_marginal(self):
        r = self.rsp
        output = np.zeros(self.graph.n_nodes)
        if not np.isfinite(self.log_partition):
            return output
        a, b = r.ptr[self.waypoints[0] : self.waypoints[0] + 2]
        lp = (
            r.log_w[a:b]
            + self.z[1, self.next_stage[0][a:b], r.cols[a:b]]
            - self.log_partition
        )
        output[r.cols[a:b]] = np.where(self.allowed[0][a:b], np.exp(lp), 0.0)
        return output

    def length_distribution(self):
        r = self.rsp
        mass = np.zeros((self.stages, self.graph.n_nodes))
        mass[0, self.waypoints[0]] = 1.0
        result = np.zeros(self.config.max_steps + 1)
        if not np.isfinite(self.log_partition):
            return result
        for p in range(self.config.max_steps + 1):
            result[p] = mass[-1, self.waypoints[-1]]
            mass[-1, self.waypoints[-1]] = 0.0
            if p == self.config.max_steps:
                break
            updated = np.zeros_like(mass)
            for q in range(self.stages):
                safe = np.where(np.isfinite(self.z[p, q]), self.z[p, q], 0.0)
                lp = r.log_w + self.z[p + 1, self.next_stage[q], r.cols] - safe[r.rows]
                probs = np.exp(lp)
                probs[~self.allowed[q] | ~np.isfinite(self.z[p, q, r.rows])] = 0.0
                np.add.at(
                    updated, (self.next_stage[q], r.cols), mass[q, r.rows] * probs
                )
            mass = updated
        return result


def route_shortest(graph, waypoints):
    """BFS on the product graph, including first-visit ordering constraints."""
    from collections import deque

    waypoints = tuple(waypoints)
    queue = deque([(waypoints[0], 1, 0)])
    seen = {(waypoints[0], 1)}
    counts = {(waypoints[0], 1): 1}
    distances = {(waypoints[0], 1): 0}
    while queue:
        node, next_index, distance = queue.popleft()
        if next_index == len(waypoints):
            return distance, counts[(node, next_index)]
        for dest in graph.adjacency.indices[
            graph.adjacency.indptr[node] : graph.adjacency.indptr[node + 1]
        ]:
            dest = int(dest)
            if dest in waypoints[next_index + 1 :]:
                continue
            stage = next_index + int(dest == waypoints[next_index])
            state = dest, stage
            if state not in seen:
                seen.add(state)
                distances[state] = distance + 1
                counts[state] = counts[(node, next_index)]
                queue.append((dest, stage, distance + 1))
            elif distances[state] == distance + 1:
                counts[state] += counts[(node, next_index)]
    return float("inf"), 0


def suffix_distances(graph, suffix):
    """Shortest admissible total distance for every start sharing this suffix."""
    from scipy.sparse.csgraph import shortest_path

    remaining = 0.0
    for j in range(len(suffix) - 1, -1, -1):
        target = suffix[j]
        forbidden = set(suffix[j + 1 :])
        adjacency = graph.adjacency.copy()
        if forbidden:
            blocked = np.isin(graph.rows, list(forbidden)) | np.isin(
                adjacency.indices, list(forbidden)
            )
            adjacency.data[blocked] = 0
            adjacency.eliminate_zeros()
        distance = shortest_path(
            adjacency, directed=False, unweighted=True, indices=target
        )
        if j == 0:
            return distance + remaining
        remaining += distance[suffix[j - 1]]

"""Finite-horizon first-hit Gibbs walks; all partition functions use log space."""

from collections import OrderedDict
from pathlib import Path
import warnings

import numpy as np
from scipy.special import logsumexp

from .graph import N_SPECIAL, PAD


class RSP:
    def __init__(self, graph, cache_dir=None):
        self.graph, self.config = graph, graph.config
        self.cache = OrderedDict()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        g, c = graph, graph.config
        self.rows, self.cols = g.rows, g.adjacency.indices
        self.ptr = g.adjacency.indptr
        preference = c.alpha * np.log(g.weights[self.cols])
        normalizer = self._row_logsumexp(preference[:, None])[:, 0]
        self.log_w = (
            preference
            - normalizer[self.rows]
            - (c.edge_cost + c.hub_cost * g.is_hub[self.cols]) / c.temperature
        )
        if c.require_simple:
            warnings.warn(
                "Simple-path rejection changes the distribution: reported walk entropy/logprob are NOT exact for accepted simple paths.",
                stacklevel=2,
            )

    def _row_logsumexp(self, values):
        # Graphs are connected and have no empty CSR rows.
        nonempty = np.flatnonzero(np.diff(self.ptr))
        output = np.full((self.graph.n_nodes, values.shape[1]), -np.inf)
        if not len(nonempty):
            return output
        maxima = np.maximum.reduceat(values, self.ptr[nonempty], axis=0)
        safe = np.zeros_like(output)
        safe[nonempty] = np.where(np.isfinite(maxima), maxima, 0.0)
        totals = np.add.reduceat(
            np.exp(values - safe[self.rows]), self.ptr[nonempty], axis=0
        )
        with np.errstate(divide="ignore"):
            output[nonempty] = safe[nonempty] + np.log(totals)
        return output

    def _row_sum(self, values):
        output = np.zeros((self.graph.n_nodes,) + values.shape[1:])
        nonempty = np.flatnonzero(np.diff(self.ptr))
        if len(nonempty):
            output[nonempty] = np.add.reduceat(values, self.ptr[nonempty], axis=0)
        return output

    def tables(self, goals, hints=None):
        """Batch goals as columns; hints map absolute path positions to NODE ids."""
        goals = np.asarray(goals, dtype=int)
        hints = hints or {}
        c, n = self.config, self.graph.n_nodes
        if any(p < 0 or p > c.max_steps or v < 0 or v >= n for p, v in hints.items()):
            raise ValueError("hint outside path/node range")
        z = np.full((c.max_steps + 1, n, len(goals)), -np.inf)
        h = np.zeros((n, len(goals)))
        columns = np.arange(len(goals))
        for p in range(c.max_steps, -1, -1):
            if p < c.max_steps:
                values = self.log_w[:, None] + z[p + 1, self.cols]
                z[p] = self._row_logsumexp(values)
                safe = np.where(np.isfinite(z[p]), z[p], 0.0)
                lp = values - safe[self.rows]
                probs = np.exp(lp)
                contributions = probs * (
                    h[self.cols] - np.where(np.isfinite(lp), lp, 0.0)
                )
                h = self._row_sum(contributions)
            # Arrival terminates the walk; future hints preclude early arrival.
            allow_stop = (c.length_mode == "at_most" or p == c.max_steps) and not any(
                q > p for q in hints
            )
            z[p, goals, columns] = 0.0 if allow_stop else -np.inf
            h[goals, columns] = 0.0
            if p in hints:
                excluded = np.arange(n) != hints[p]
                z[p, excluded] = -np.inf
                h[excluded] = 0.0
        return z, h / np.log(2)

    def table(self, goal, hints=None):
        key = (int(goal), tuple(sorted((hints or {}).items())))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        zpath = (
            self.cache_dir / f"goal_{goal}_z.npy"
            if self.cache_dir and not hints
            else None
        )
        hpath = self.cache_dir / f"goal_{goal}_h.npy" if zpath else None
        if zpath and zpath.exists() and hpath.exists():
            result = np.load(zpath, mmap_mode="r"), np.load(hpath, mmap_mode="r")
        else:
            z, h = self.tables([goal], hints)
            result = z[:, :, 0], h[:, 0]
            if zpath:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                np.save(zpath, result[0])
                np.save(hpath, result[1])
        self.cache[key] = result
        while len(self.cache) > self.config.cache_goals:
            self.cache.popitem(last=False)
        return result

    def prefetch(self, goals):
        z, h = self.tables(goals)
        for b, goal in enumerate(goals):
            # Copies prevent one LRU entry retaining the entire goal batch.
            self.cache[(int(goal), ())] = z[:, :, b].copy(), h[:, b].copy()
        while len(self.cache) > self.config.cache_goals:
            self.cache.popitem(last=False)

    def sample(self, start, goal, rng, hints=None):
        z, _ = self.table(goal, hints)
        if not np.isfinite(z[0, start]):
            raise ValueError("prompt/hints have zero probability at this horizon")
        for _ in range(self.config.max_sampling_attempts):
            path = [int(start)]
            for p in range(self.config.max_steps):
                i = path[-1]
                if i == goal:
                    break
                a, b = self.ptr[i : i + 2]
                scores = self.log_w[a:b] + z[p + 1, self.cols[a:b]]
                probabilities = np.exp(scores - logsumexp(scores))
                path.append(int(rng.choice(self.cols[a:b], p=probabilities)))
            if not self.config.require_simple or len(path) == len(set(path)):
                return path
        raise RuntimeError("simple-path rejection exhausted max_sampling_attempts")

    def true_logprob(self, start, goal, path, hints=None):
        from .verify import verify_path

        result = verify_path(
            self.graph,
            start,
            goal,
            path,
            max_steps=self.config.max_steps,
            require_simple=self.config.require_simple,
            hints=hints,
        )
        if not result.valid or (
            self.config.length_mode == "exact"
            and result.n_steps != self.config.max_steps
        ):
            return -np.inf
        nodes = np.asarray([int(t) - N_SPECIAL for t in path if t != PAD])
        keys = nodes[:-1] * self.graph.n_nodes + nodes[1:]
        idx = np.searchsorted(self.graph.edge_keys, keys)
        z, _ = self.table(goal, hints)
        if not np.isfinite(z[0, start]):
            return -np.inf
        return float(self.log_w[idx].sum() - z[0, start])

    def cond_entropy_bits(self, start, goal, hints=None):
        z, h = self.table(goal, hints)
        return float(h[start]) if np.isfinite(z[0, start]) else float("nan")

    def first_step_marginal(self, start, goal, hints=None):
        z, _ = self.table(goal, hints)
        output = np.zeros(self.graph.n_nodes)
        if start == goal or not np.isfinite(z[0, start]):
            return output
        a, b = self.ptr[start : start + 2]
        output[self.cols[a:b]] = np.exp(
            self.log_w[a:b] + z[1, self.cols[a:b]] - z[0, start]
        )
        return output

    def length_distribution(self, start, goal, hints=None):
        z, _ = self.table(goal, hints)
        mass = np.zeros(self.graph.n_nodes)
        mass[start] = 1.0
        lengths = np.zeros(self.config.max_steps + 1)
        if not np.isfinite(z[0, start]):
            return lengths
        for p in range(self.config.max_steps + 1):
            lengths[p] = mass[goal]
            mass[goal] = 0.0
            if p == self.config.max_steps:
                break
            safe = np.where(np.isfinite(z[p]), z[p], 0.0)
            lp = self.log_w + z[p + 1, self.cols] - safe[self.rows]
            probability = np.exp(lp)
            probability[~np.isfinite(z[p, self.rows])] = 0.0
            mass = np.bincount(
                self.cols, weights=mass[self.rows] * probability, minlength=len(mass)
            )
        return lengths

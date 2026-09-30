"""Hidden core-token chains with frozen low-rank preferences and RSP bridges.

There is no per-pair random score table. Pair preferences are a product of two
N-by-r factors. A node can be a core occurrence in one place and a connector in
another; only sidecar arrays record those roles.
"""

from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .data import _Writer
from .graph import N_SPECIAL, SEP, generate_graph, save_npz
from .rsp import RSP


@dataclass(frozen=True)
class LatentCoreConfig:
    n_cores: int = 8
    bridge_max_steps: int = 10
    min_distance: int = 2
    max_distance: int = 8
    distance_target: float = 4.0
    distance_width: float = 2.0
    baseline_log_std: float = 1.0
    bridge_avoidance: float = 1.5
    bridge_pseudocount: float = 1.0
    pilot_paths: int = 2048
    pilot_starts_per_goal: int = 32
    correlation_rank: int = 8
    correlation_strength: float = 1.0
    topic_dim: int = 4
    topic_strength: float = 0.75
    topic_persistence: float = 0.95
    distance_cache_size: int = 32
    max_attempts_per_example: int = 1000
    seed: int = 0

    @property
    def seq_len(self):
        return 2 + (self.n_cores - 1) * self.bridge_max_steps

    def validate(self):
        integer_fields = (
            "n_cores",
            "bridge_max_steps",
            "min_distance",
            "max_distance",
            "pilot_paths",
            "pilot_starts_per_goal",
            "correlation_rank",
            "topic_dim",
            "distance_cache_size",
            "max_attempts_per_example",
        )
        if any(
            not isinstance(getattr(self, k), int) or getattr(self, k) < 1
            for k in integer_fields
        ):
            raise ValueError("latent counts and distances must be positive integers")
        if (
            self.n_cores < 2
            or not 1 <= self.min_distance <= self.max_distance <= self.bridge_max_steps
        ):
            raise ValueError(
                "require n_cores >= 2 and 1 <= min_distance <= max_distance <= bridge_max_steps"
            )
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("latent config values must be finite")
        if min(self.distance_width, self.bridge_pseudocount) <= 0:
            raise ValueError("distance_width and bridge_pseudocount must be positive")
        if (
            min(
                self.baseline_log_std,
                self.bridge_avoidance,
                self.correlation_strength,
                self.topic_strength,
            )
            < 0
        ):
            raise ValueError("preference strengths must be nonnegative")
        if not 0 <= self.topic_persistence < 1:
            raise ValueError("topic_persistence must lie in [0,1)")
        return self


def bridge_rsp(graph, config):
    c = replace(
        graph.config,
        max_steps=config.bridge_max_steps,
        length_mode="at_most",
        require_simple=False,
        n_hints=0,
    )
    return RSP(replace(graph, config=c))


def estimate_bridge_visitation(graph, config):
    """Neutral uniform endpoint pilot; count ONLY interior node occurrences."""
    rng = np.random.default_rng(config.seed + 1103)
    rsp = bridge_rsp(graph, config)
    counts = np.zeros(graph.n_nodes, dtype=np.int64)
    samples, attempts = 0, 0
    while samples < config.pilot_paths:
        attempts += 1
        if attempts > config.pilot_paths * config.max_attempts_per_example:
            raise ValueError("no feasible calibration bridges in the distance band")
        goal = int(rng.integers(graph.n_nodes))
        distances = graph.distances(goal)
        starts = np.flatnonzero(
            (distances >= config.min_distance) & (distances <= config.max_distance)
        )
        if not len(starts):
            continue
        take = min(config.pilot_starts_per_goal, config.pilot_paths - samples)
        for start in rng.choice(starts, take):
            path = rsp.sample(int(start), goal, rng)
            np.add.at(counts, path[1:-1], 1)
            samples += 1
    return counts


class CorePrior:
    """Frozen node fields; memory O(N*(rank+topic_dim)), never O(N**2)."""

    def __init__(self, graph, config, bridge_counts, *, fields=None):
        self.graph, self.config = graph, config.validate()
        self.distance_cache = OrderedDict()
        self.bridge_counts = np.asarray(bridge_counts, dtype=np.int64)
        if (
            self.bridge_counts.shape != (graph.n_nodes,)
            or (self.bridge_counts < 0).any()
        ):
            raise ValueError("invalid bridge counts")
        if fields is None:
            rng = np.random.default_rng(config.seed + 1109)
            self.baseline_logits = rng.normal(
                0.0, config.baseline_log_std, graph.n_nodes
            )
            self.source_factors = rng.normal(
                size=(graph.n_nodes, config.correlation_rank)
            )
            self.target_factors = rng.normal(
                size=(graph.n_nodes, config.correlation_rank)
            )
            self.topic_loadings = rng.normal(size=(graph.n_nodes, config.topic_dim))
        else:
            for name in (
                "baseline_logits",
                "source_factors",
                "target_factors",
                "topic_loadings",
            ):
                setattr(self, name, np.asarray(fields[name]))
        frequency = self.bridge_counts + config.bridge_pseudocount
        self.bridge_frequency = frequency / frequency.sum()
        # Smooth, finite penalty: even a high-traffic router retains positive mass.
        self.log_base = self.baseline_logits - config.bridge_avoidance * np.log1p(
            graph.n_nodes * self.bridge_frequency
        )
        self.base_probabilities = np.exp(self.log_base - logsumexp(self.log_base))

    def distances(self, source):
        source = int(source)
        if source not in self.distance_cache:
            self.distance_cache[source] = self.graph.distances(source)
        self.distance_cache.move_to_end(source)
        while len(self.distance_cache) > self.config.distance_cache_size:
            self.distance_cache.popitem(last=False)
        return self.distance_cache[source]

    def probabilities(self, source, topic):
        """Return pre-filter law, actual conditional law, and distance array.

        pre-filter includes baseline, soft connector avoidance and topic/pair
        factors; post-filter additionally includes distance weights/restrictions.
        """
        c = self.config
        topic = np.asarray(topic)
        if topic.shape != (c.topic_dim,):
            raise ValueError("topic has wrong dimension")
        scores = self.log_base + c.topic_strength * (
            self.topic_loadings @ topic
        ) / np.sqrt(c.topic_dim)
        if source is not None:
            scores = scores + c.correlation_strength * (
                self.target_factors @ self.source_factors[source]
            ) / np.sqrt(c.correlation_rank)
        before = np.exp(scores - logsumexp(scores))
        if source is None:
            return before, before.copy(), None
        distances = self.distances(source)
        allowed = (distances >= c.min_distance) & (distances <= c.max_distance)
        scores = np.where(
            allowed,
            scores - 0.5 * ((distances - c.distance_target) / c.distance_width) ** 2,
            -np.inf,
        )
        normalizer = logsumexp(scores)
        if not np.isfinite(normalizer):
            raise ValueError("core has no feasible successor in the distance band")
        return before, np.exp(scores - normalizer), distances

    def save(self, path):
        save_npz(
            path,
            config=json.dumps(asdict(self.config), sort_keys=True),
            bridge_counts=self.bridge_counts,
            bridge_frequency=self.bridge_frequency,
            baseline_logits=self.baseline_logits,
            base_probabilities=self.base_probabilities,
            source_factors=self.source_factors,
            target_factors=self.target_factors,
            topic_loadings=self.topic_loadings,
        )

    @classmethod
    def load(cls, graph, path):
        with np.load(path) as arrays:
            c = LatentCoreConfig(**json.loads(str(arrays["config"])))
            return cls(graph, c, arrays["bridge_counts"], fields=arrays)

    def draw_cores(self, rng):
        c = self.config
        topic = rng.normal(size=c.topic_dim)
        cores, topics, logprob, distances = [], [], [], []
        before_total = np.zeros(self.graph.n_nodes)
        after_total = np.zeros(self.graph.n_nodes)
        for index in range(c.n_cores):
            if index:
                topic = c.topic_persistence * topic + np.sqrt(
                    1 - c.topic_persistence**2
                ) * rng.normal(size=c.topic_dim)
            before, after, distance = self.probabilities(
                cores[-1] if cores else None, topic
            )
            node = int(rng.choice(self.graph.n_nodes, p=after))
            if distance is not None:
                distances.append(int(distance[node]))
            cores.append(node)
            topics.append(topic.copy())
            logprob.append(float(np.log(after[node])))
            before_total += before
            after_total += after
        return dict(
            cores=cores,
            topics=topics,
            core_logprob_given_topics=logprob,
            distances=distances,
            pre_filter_expected=before_total,
            post_filter_expected=after_total,
        )


def draw_document(prior, rsp, rng):
    sample = prior.draw_cores(rng)
    cores = sample["cores"]
    path, positions, entropies, logprobs = [cores[0]], [0], [], []
    for start, goal in zip(cores, cores[1:]):
        bridge = rsp.sample(start, goal, rng)
        entropies.append(rsp.cond_entropy_bits(start, goal))
        # The sampled bridge is valid by construction; use its edge weights
        # directly rather than computing shortest-path verifier metrics again.
        keys = np.asarray(bridge[:-1]) * prior.graph.n_nodes + bridge[1:]
        indices = np.searchsorted(prior.graph.edge_keys, keys)
        logprobs.append(float(rsp.log_w[indices].sum() - rsp.table(goal)[0][0, start]))
        path.extend(bridge[1:])
        positions.append(len(path) - 1)
    sample.update(
        path=path,
        positions=positions,
        bridge_entropy_bits=entropies,
        bridge_logprob=logprobs,
    )
    return sample


def build_latent_dataset(graph_config, latent_config, output):
    c = latent_config.validate()
    if (
        graph_config.n_hints
        or graph_config.require_simple
        or graph_config.length_mode != "at_most"
    ):
        raise ValueError(
            "latent-core mode requires n_hints=0, require_simple=false, length_mode=at_most"
        )
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("output directory must be empty")
    graph = generate_graph(graph_config)
    graph.save(root / "graph.npz")
    pilot = estimate_bridge_visitation(graph, c)
    prior = CorePrior(graph, c, pilot)
    prior.save(root / "core_prior.npz")
    rsp = bridge_rsp(graph, c)
    quotas = dict(
        train=graph_config.n_train,
        validation=graph_config.n_val,
        test=graph_config.n_test,
    )
    # Latent language modeling holds out complete observed documents, not pairs.
    # Same core pairs may recur across splits, as intended for a shared grammar.
    owner = {}
    node_counts = np.zeros(graph.n_nodes, dtype=np.int64)
    edge_counts = np.zeros(len(graph.edge_keys), dtype=np.int64)
    split_stats = {}
    definitions = [
        ("cores", np.int32, (c.n_cores,)),
        ("positions", np.int32, (c.n_cores,)),
        ("topics", np.float64, (c.n_cores, c.topic_dim)),
        ("core_logprob_given_topics", np.float64, (c.n_cores,)),
        ("distances", np.int32, (c.n_cores - 1,)),
        ("bridge_entropy_bits", np.float64, (c.n_cores - 1,)),
        ("bridge_logprob", np.float64, (c.n_cores - 1,)),
    ]
    for split_index, (split, count) in enumerate(quotas.items()):
        rng = np.random.default_rng(c.seed + 1123 + split_index)
        writer = _Writer(root, split, c.seq_len)
        arrays = {
            key: np.lib.format.open_memmap(
                root / f"{split}_latent_{key}.npy",
                mode="w+",
                dtype=dtype,
                shape=(count,) + shape,
            )
            for key, dtype, shape in definitions
        }
        before, after = np.zeros(graph.n_nodes), np.zeros(graph.n_nodes)
        rejected, infeasible = 0, 0
        for index in range(count):
            for _ in range(c.max_attempts_per_example):
                try:
                    sample = draw_document(prior, rsp, rng)
                except ValueError as error:
                    if (
                        str(error)
                        != "core has no feasible successor in the distance band"
                    ):
                        raise
                    infeasible += 1
                    continue
                fingerprint = hashlib.sha256(
                    np.asarray(sample["path"], dtype="<i4").tobytes()
                ).digest()
                if fingerprint in owner and owner[fingerprint] != split:
                    rejected += 1
                    continue
                owner[fingerprint] = split
                break
            else:
                raise RuntimeError(
                    "failed to obtain a feasible, split-disjoint document"
                )
            row = np.zeros(c.seq_len, dtype=np.int32)
            row[0] = SEP  # neutral BOS, never an interior core marker
            row[1 : 1 + len(sample["path"])] = np.asarray(sample["path"]) + N_SPECIAL
            mask = np.ones(c.seq_len, dtype=np.uint8)
            mask[0] = 0
            writer.append(
                row,
                mask,
                [
                    sum(sample["bridge_logprob"]),
                    sum(sample["bridge_entropy_bits"]),
                    sum(sample["distances"]),
                    0,
                ],
            )
            for key, array in arrays.items():
                array[index] = sample[key]
            before += sample["pre_filter_expected"]
            after += sample["post_filter_expected"]
            if split == "train":
                path = np.asarray(sample["path"])
                np.add.at(node_counts, path, 1)
                keys = path[:-1] * graph.n_nodes + path[1:]
                np.add.at(edge_counts, np.searchsorted(graph.edge_keys, keys), 1)
        writer.finish()
        for array in arrays.values():
            array.flush()
        arrays.clear()
        save_npz(
            root / f"{split}_latent_prior_expectations.npz",
            pre_filter=before,
            post_filter=after,
        )
        split_stats[split] = dict(
            examples=count,
            rejected_cross_split_duplicates=rejected,
            rejected_infeasible_chains=infeasible,
        )
    np.save(root / "train_node_counts.npy", node_counts)
    np.save(root / "train_edge_counts.npy", edge_counts)
    metadata = dict(
        format_version=2,
        task_mode="latent_core",
        config=graph_config.to_dict(),
        latent_config=asdict(c),
        seq_len=c.seq_len,
        special_tokens={"PAD": 0, "BOS": SEP, "MASK": 2},
        counts=quotas,
        split_statistics=split_stats,
        model_fields=["input_ids", "attention_mask"],
        latent_fields=[key for key, _, _ in definitions],
        core_positions="zero-based within unmarked node sequence, excluding BOS",
        metric_columns=[
            "segmented_bridge_logprob_given_cores",
            "sum_bridge_entropy_bits_given_cores",
            "sum_shortest_bridge_distances",
            "unused",
        ],
        entropy_conditioning="independent first-hit bridges given latent endpoint pairs; not visible-sequence entropy or NLL",
        split_policy="no identical finished node sequences across splits; latent core combinations may recur",
        ignored_conditional_mode_knobs=[
            "n_waypoints",
            "max_steps",
            "min_dist",
            "max_dist",
            "endpoint_alpha",
            "min_edge_count",
            "holdout_cluster_pairs",
            "starts_per_goal",
            "goal_batch_size",
        ],
        bridge_pilot=dict(
            paths=c.pilot_paths,
            interior_visits=int(pilot.sum()),
            observed_nodes=int((pilot > 0).sum()),
            endpoint_prior="uniform goals, uniform feasible starts",
        ),
        require_simple=False,
        exact_visible_sequence_likelihood=False,
    )
    (root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata

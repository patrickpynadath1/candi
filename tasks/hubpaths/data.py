"""Pair-disjoint streaming dataset preparation and the local task adapter."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .graph import N_SPECIAL, PAD, SEP, MASK, generate_graph
from .rsp import RSP
from .routes import RouteDistribution, suffix_distances


class HubPathsTokenizer:
    pad_token_id, sep_token_id, mask_token_id = PAD, SEP, MASK
    bos_token_id = sep_token_id
    eos_token_id = pad_token_id
    pad_token, sep_token, mask_token = "<PAD>", "<SEP>", "<MASK>"
    bos_token, eos_token = "<SEP>", "<PAD>"
    prompt_len = 3

    def __init__(
        self, n_nodes, max_steps=77, n_waypoints=3, *, seq_len=None, prompt_len=None
    ):
        self.vocab_size = n_nodes + N_SPECIAL
        self.prompt_len = n_waypoints + 1 if prompt_len is None else prompt_len
        self.seq_len = self.total_seq_len = (
            max_steps + n_waypoints + 2 if seq_len is None else seq_len
        )

    def __len__(self):
        return self.vocab_size

    def decode(self, tokens, skip_special_tokens=False):
        names = {PAD: self.pad_token, SEP: self.sep_token, MASK: self.mask_token}
        return " ".join(
            names.get(int(t), str(int(t) - N_SPECIAL))
            for t in tokens
            if not skip_special_tokens or int(t) >= N_SPECIAL
        )

    def batch_decode(self, sequences, skip_special_tokens=False):
        return [self.decode(s, skip_special_tokens) for s in sequences]


def encode(waypoints, path, config, rng):
    prefix = len(waypoints) + 1
    target = np.zeros(config.seq_len, dtype=np.int32)
    target[: prefix - 1] = np.asarray(waypoints) + N_SPECIAL
    target[prefix - 1] = SEP
    target[prefix : prefix + len(path)] = np.asarray(path) + N_SPECIAL
    mask = np.ones(config.seq_len, dtype=np.uint8)
    mask[:prefix] = 0
    interior = np.arange(1, len(path) - 1)
    positions = rng.choice(
        interior, size=min(config.n_hints, len(interior)), replace=False
    )
    mask[positions + prefix] = 0
    return target, mask


def hints_from_row(row, mask, n_waypoints=3):
    prefix = n_waypoints + 1
    return {
        int(i): int(row[i + prefix]) - N_SPECIAL
        for i in np.flatnonzero(mask[prefix:] == 0)
    }


class PairSplits:
    """Stable unordered-pair hashing reserves pairs before any sampling/top-ups."""

    def __init__(self, graph):
        self.graph, self.c = graph, graph.config
        rng = np.random.default_rng(self.c.seed + 1)
        pairs = [
            (a, b)
            for a in range(self.c.n_clusters)
            for b in range(a + 1, self.c.n_clusters)
        ]
        self.ood = {
            pairs[i]
            for i in rng.choice(len(pairs), self.c.holdout_cluster_pairs, replace=False)
        }
        total = max(1, self.c.n_train + self.c.n_val + self.c.n_test)
        # Reserve enough unique evaluation pairs even when n_train is very large.
        self.val_fraction = (
            min(0.2, max(0.05, self.c.n_val / total)) if self.c.n_val else 0.0
        )
        self.test_fraction = (
            min(0.2, max(0.05, self.c.n_test / total)) if self.c.n_test else 0.0
        )

    def split(self, start, goal):
        clusters = tuple(
            sorted((int(self.graph.cluster[start]), int(self.graph.cluster[goal])))
        )
        if clusters in self.ood:
            return "test_ood"
        a, b = sorted((int(start), int(goal)))
        payload = f"{self.c.seed}:{a}:{b}".encode()
        value = (
            int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
            / 2**64
        )
        if value < self.val_fraction:
            return "validation"
        if value < self.val_fraction + self.test_fraction:
            return "test"
        return "train"


class _Writer:
    def __init__(self, root, split, width):
        self.root, self.split, self.width = root, split, width
        self.count = 0
        self.files = {
            k: (root / f".{split}_{k}.raw").open("wb")
            for k in ("tokens", "mask", "metrics")
        }

    def append(self, row, mask, metrics):
        self.files["tokens"].write(np.asarray(row, dtype=np.int32).tobytes())
        self.files["mask"].write(np.asarray(mask, dtype=np.uint8).tobytes())
        self.files["metrics"].write(np.asarray(metrics, dtype=np.float64).tobytes())
        self.count += 1

    def finish(self):
        for key, dtype, width in [
            ("tokens", np.int32, self.width),
            ("mask", np.uint8, self.width),
            ("metrics", np.float64, 4),
        ]:
            self.files[key].close()
            source = self.root / f".{self.split}_{key}.raw"
            output = np.lib.format.open_memmap(
                self.root / f"{self.split}_{key}.npy",
                mode="w+",
                dtype=dtype,
                shape=(self.count, width),
            )
            if self.count:
                raw = np.memmap(
                    source, mode="r", dtype=dtype, shape=(self.count, width)
                )
                for at in range(0, self.count, 8192):
                    output[at : at + 8192] = raw[at : at + 8192]
                del raw
            del output
            source.unlink()


def build_dataset(config, output):
    c = config.validate()
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError(f"output directory must be empty: {root}")
    graph = generate_graph(c)
    graph.save(root / "graph.npz")
    rsp, splits = RSP(graph), PairSplits(graph)
    rng = np.random.default_rng(c.seed + 2)
    quotas = {
        "train": c.n_train,
        "validation": c.n_val,
        "test": c.n_test,
        "test_ood": c.n_test if c.holdout_cluster_pairs else 0,
    }
    writers = {name: _Writer(root, name, c.seq_len) for name in quotas}
    edge_counts = np.zeros(len(graph.edge_keys), dtype=np.int64)
    node_counts = np.zeros(graph.n_nodes, dtype=np.int64)
    endpoint = c.endpoint_alpha * np.log(graph.weights)
    endpoint = np.exp(endpoint - endpoint.max())
    endpoint /= endpoint.sum()
    eval_seen = {name: set() for name in quotas if name != "train"}

    def route_split(waypoints):
        # Endpoint reservations still exclude reversed endpoint pairs across splits.
        split = splits.split(waypoints[0], waypoints[-1])
        if split != "test_ood":
            clusters = graph.cluster[list(waypoints)]
            if any(
                tuple(sorted((int(a), int(b)))) in splits.ood
                for a in clusters
                for b in clusters
            ):
                return None
        return split

    def append(split, distribution, path, distance, topup=False):
        waypoints = distribution.waypoints
        row, mask = encode(waypoints, path, c, rng)
        if split == "train":
            metrics = [np.nan, np.nan, distance, int(topup)]
            keys = np.asarray(path[:-1]) * graph.n_nodes + path[1:]
            np.add.at(edge_counts, np.searchsorted(graph.edge_keys, keys), 1)
            np.add.at(node_counts, path, 1)
        else:
            hints = hints_from_row(row, mask, c.n_waypoints)
            conditional = (
                RouteDistribution(rsp, waypoints, hints) if hints else distribution
            )
            metrics = [
                conditional.true_logprob(row[c.n_waypoints + 1 :]),
                conditional.cond_entropy_bits(),
                distance,
                0,
            ]
            # Persist a bounded number of suffix tables; all others regenerate lazily.
            cache = root / "goal_cache"
            cache.mkdir(exist_ok=True)
            key = "_".join(map(str, waypoints[1:]))
            destination = cache / f"suffix_{key}.npy"
            if (
                not destination.exists()
                and len(list(cache.glob("suffix_*_entropy.npy"))) < c.cache_goals
            ):
                np.save(destination, distribution.z)
                np.save(cache / f"suffix_{key}_entropy.npy", distribution.entropies)
        writers[split].append(row, mask, metrics)

    stalled = 0
    while any(writers[k].count < q for k, q in quotas.items()):
        before = sum(w.count for w in writers.values())
        candidates = []
        for _ in range(c.goal_batch_size):
            # Draw the final goal first so its marginal is proportional to w**endpoint_alpha.
            selected = rng.choice(
                graph.n_nodes, size=c.n_waypoints - 1, p=endpoint, replace=False
            )
            suffix = tuple(int(v) for v in np.concatenate((selected[1:], selected[:1])))
            distances = suffix_distances(graph, suffix)
            starts = np.flatnonzero(
                (distances >= c.min_dist) & (distances <= c.max_dist)
            )
            starts = starts[~np.isin(starts, suffix)]
            if len(starts):
                candidates.append((suffix, distances, starts))
        bases = RouteDistribution.batch(
            rsp, [(int(starts[0]),) + suffix for suffix, _, starts in candidates]
        )
        for base, (_, distances, starts) in zip(bases, candidates):
            starts = starts[np.isfinite(base.z[0, 0, starts])]
            if len(starts):
                for start in rng.choice(starts, size=c.starts_per_goal, replace=True):
                    distribution = base.with_start(int(start))
                    split = route_split(distribution.waypoints)
                    if split is None or writers[split].count >= quotas[split]:
                        continue
                    route = distribution.waypoints
                    key = min(route, route[::-1])
                    if split != "train" and key in eval_seen[split]:
                        continue
                    path = distribution.sample(rng)
                    append(split, distribution, path, distances[start])
                    if split != "train":
                        eval_seen[split].add(key)
        after = sum(w.count for w in writers.values())
        stalled = stalled + 1 if before == after else 0
        if stalled >= min(c.max_sampling_attempts, 1000):
            raise RuntimeError(
                "cannot fill split quotas: reduce eval sizes or widen distance band/graph; partial data retained"
            )
    # Keep top-ups on the same conditional law; alter only prompt selection.
    reverse = np.searchsorted(
        graph.edge_keys, graph.adjacency.indices * graph.n_nodes + graph.rows
    )
    attempts = 0
    if c.n_train and c.min_edge_count:
        for edge in np.flatnonzero(graph.rows < graph.adjacency.indices):
            a, b = int(graph.rows[edge]), int(graph.adjacency.indices[edge])
            edge_attempts = 0
            while edge_counts[edge] + edge_counts[reverse[edge]] < c.min_edge_count:
                attempts += 1
                edge_attempts += 1
                if attempts > c.max_sampling_attempts or edge_attempts > 100:
                    break
                # Begin with the undercovered edge, then extend a distinct-node
                # waypoint list along graph edges. Sampling itself is unforced.
                points = [a, b]
                while len(points) < c.n_waypoints:
                    neighbors = graph.adjacency.indices[
                        graph.adjacency.indptr[points[-1]] : graph.adjacency.indptr[
                            points[-1] + 1
                        ]
                    ]
                    candidates = [int(v) for v in neighbors if v not in points]
                    if not candidates:
                        break
                    points.append(int(rng.choice(candidates)))
                if len(points) != c.n_waypoints or route_split(points) != "train":
                    continue
                distribution = RouteDistribution(rsp, points)
                if not np.isfinite(distribution.log_partition):
                    continue
                path = distribution.sample(rng)
                append("train", distribution, path, c.n_waypoints - 1, True)
            if attempts > c.max_sampling_attempts:
                break
    for writer in writers.values():
        writer.finish()
    counts = edge_counts + edge_counts[reverse]
    if c.n_train and np.any(counts < c.min_edge_count):
        import warnings

        warnings.warn(
            "Coverage target not reached within the top-up budget; see metadata.coverage.edges_below_target.",
            stacklevel=2,
        )
    metadata = {
        "format_version": 1,
        "config": c.to_dict(),
        "seq_len": c.seq_len,
        "special_tokens": {"PAD": PAD, "SEP": SEP, "MASK": MASK},
        "counts": {k: w.count for k, w in writers.items()},
        "heldout_cluster_pairs": sorted(splits.ood),
        "metric_columns": [
            "true_logprob",
            "cond_entropy_bits",
            "shortest_dist",
            "coverage_topup",
        ],
        "coverage": {
            "undirected": True,
            "min_edge_count": int(counts.min()),
            "edges_below_target": int((counts < c.min_edge_count).sum() // 2),
            "topup_attempts": attempts,
        },
        "exact_walk_statistics": not c.require_simple,
        "hint_conditioning": "fixed-position constraints; does not model the random hint-selection mechanism",
        "eval_routes_unique_up_to_reversal": True,
        "waypoint_semantics": "distinct nodes, ordered first visits, global horizon",
    }
    np.save(root / "train_edge_counts.npy", edge_counts)
    np.save(root / "train_node_counts.npy", node_counts)
    (root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


class HubPathsDataset(Dataset):
    def __init__(self, directory, split="train", view="infilling"):
        self.directory, self.view = Path(directory), view
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        self.tokens = np.load(self.directory / f"{split}_tokens.npy", mmap_mode="r")
        self.masks = np.load(self.directory / f"{split}_mask.npy", mmap_mode="r")
        if view not in {"infilling", "ar"}:
            raise ValueError("view must be infilling or ar")
        if view == "ar" and self.metadata["config"]["n_hints"]:
            raise ValueError(
                "AR uses a prefix prompt: generate n_hints=0 data for the AR view"
            )

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor(self.tokens[index], dtype=torch.long),
            "attention_mask": torch.tensor(self.masks[index], dtype=torch.long),
        }

    def infilling_input(self, index):
        item = self[index]
        return item["input_ids"].masked_fill(item["attention_mask"].bool(), MASK)


def get_tokenizer(config):
    metadata = json.loads(
        (Path(config.data.prepared_dir) / "metadata.json").read_text()
    )
    c = metadata["config"]
    if config.model.length != metadata["seq_len"]:
        raise ValueError(f"model.length must equal saved seq_len={metadata['seq_len']}")
    latent = metadata.get("task_mode") in {"latent_core", "grammar_core"}
    tokenizer = HubPathsTokenizer(
        c["n_nodes"],
        c["max_steps"],
        c["n_waypoints"],
        seq_len=metadata["seq_len"],
        prompt_len=1 if latent else None,
    )
    if latent:
        tokenizer.sep_token = tokenizer.bos_token = "<BOS>"
    return tokenizer


def load_dataset(config, tokenizer, split):
    return HubPathsDataset(
        config.data.prepared_dir,
        split,
        "ar" if config.algo.name == "ar" else "infilling",
    )

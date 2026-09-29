"""Realized hidden-core statistics, including sparse (not N-by-N) pair counts."""

from collections import Counter
import json
from pathlib import Path

import numpy as np

from .graph import Graph, N_SPECIAL, PAD
from .latent import CorePrior


def mutual_information(pairs):
    """Plug-in MI in bits, with positional marginals inferred from directed pairs."""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    total = len(pairs)
    if not total:
        return 0.0
    joint = Counter(map(tuple, pairs))
    left, right = Counter(pairs[:, 0]), Counter(pairs[:, 1])
    return float(
        sum(
            count / total * np.log2(count * total / (left[a] * right[b]))
            for (a, b), count in joint.items()
        )
    )


def _distribution(counts):
    counts = np.asarray(counts, dtype=np.float64)
    return counts / max(1.0, counts.sum())


def latent_report(directory, output, split="train", max_examples=None):
    from .diagnostics import _entropy, _hist, _plot

    root, out = Path(directory), Path(output)
    graph = Graph.load(root / "graph.npz")
    prior = CorePrior.load(graph, root / "core_prior.npz")
    config = prior.config
    cores = np.load(root / f"{split}_latent_cores.npy", mmap_mode="r")
    count = len(cores) if max_examples is None else min(len(cores), max_examples)
    cores = np.asarray(cores[:count])
    positions = np.load(root / f"{split}_latent_positions.npy", mmap_mode="r")[:count]
    topics = np.load(root / f"{split}_latent_topics.npy", mmap_mode="r")[:count]
    distances = np.load(root / f"{split}_latent_distances.npy", mmap_mode="r")[:count]
    entropies = np.load(
        root / f"{split}_latent_bridge_entropy_bits.npy", mmap_mode="r"
    )[:count]
    tokens = np.load(root / f"{split}_tokens.npy", mmap_mode="r")[:count]
    core_counts = np.bincount(cores.ravel(), minlength=graph.n_nodes)
    connector_counts = np.zeros(graph.n_nodes, dtype=np.int64)
    for row, anchors in zip(tokens, positions):
        nodes = row[1:][row[1:] != PAD] - N_SPECIAL
        interior = np.ones(len(nodes), dtype=bool)
        interior[anchors] = False
        np.add.at(connector_counts, nodes[interior], 1)
    realized = _distribution(core_counts)
    # Recompute for partial reports; frozen per-split expectation sums cover only
    # complete splits. Evaluated contexts are the actual sampled latent topics.
    if count == len(np.load(root / f"{split}_latent_cores.npy", mmap_mode="r")):
        with np.load(root / f"{split}_latent_prior_expectations.npz") as a:
            before, after = (
                _distribution(a["pre_filter"]),
                _distribution(a["post_filter"]),
            )
    else:
        before, after = np.zeros(graph.n_nodes), np.zeros(graph.n_nodes)
        for row, topic_rows in zip(cores, topics):
            for i, topic in enumerate(topic_rows):
                pre, post, _ = prior.probabilities(
                    int(row[i - 1]) if i else None, topic
                )
                before += pre
                after += post
        before, after = _distribution(before), _distribution(after)
    pair_array = np.stack((cores[:, :-1], cores[:, 1:]), axis=-1).reshape(-1, 2)
    pairs = Counter(map(tuple, pair_array))
    observed_mi = mutual_information(pair_array)
    rng = np.random.default_rng(config.seed + 2003)
    null_mi = []
    for _ in range(8):
        # Shuffle destinations within each core position, preserving per-position
        # marginals while removing within-document dependency.
        shuffled = cores[:, 1:].copy()
        for j in range(shuffled.shape[1]):
            rng.shuffle(shuffled[:, j])
        null_pairs = np.stack((cores[:, :-1], shuffled), axis=-1).reshape(-1, 2)
        null_mi.append(mutual_information(null_pairs))
    sorted_baseline = np.exp(prior.baseline_logits - np.max(prior.baseline_logits))
    baseline = _distribution(sorted_baseline)
    with (out / "core_frequencies.tsv").open("w") as f:
        f.write(
            "node_id\ttoken_id\tcore_count\tcore_frequency\tconnector_count\tpilot_interior_count\tbaseline_probability\tsoft_base_probability\tpre_filter_expected\tpost_filter_expected\n"
        )
        for node in range(graph.n_nodes):
            f.write(
                f"{node}\t{node + N_SPECIAL}\t{core_counts[node]}\t{realized[node]:.12g}\t{connector_counts[node]}\t{prior.bridge_counts[node]}\t{baseline[node]:.12g}\t{prior.base_probabilities[node]:.12g}\t{before[node]:.12g}\t{after[node]:.12g}\n"
            )
    with (out / "consecutive_core_counts.tsv").open("w") as f:
        f.write("source_node\ttarget_node\tcount\n")
        for (a, b), value in sorted(pairs.items()):
            f.write(f"{a}\t{b}\t{value}\n")
    for name, values in [
        ("core_distance", distances.ravel()),
        ("bridge_entropy_bits", entropies.ravel()),
    ]:
        histogram = _hist(values)
        centers = (
            np.asarray(histogram["bin_edges"][:-1]) + histogram["bin_edges"][1:]
        ) / 2
        _plot(
            out / f"{name}.svg",
            centers,
            histogram["counts"],
            name.replace("_", " ").title(),
            "value",
            "count",
        )
    _plot(
        out / "core_prior_vs_realized.svg",
        after,
        realized,
        "Core frequency: filtered prior vs realized",
        "filtered probability",
        "realized probability",
        True,
    )
    _plot(
        out / "bridge_visitation_vs_core_frequency.svg",
        prior.bridge_frequency,
        realized,
        "Pilot bridge visitation vs realized core frequency",
        "pilot bridge frequency",
        "core frequency",
        True,
    )
    by_distance = []
    for distance in np.unique(distances):
        mask = distances == distance
        by_distance.append(
            dict(
                distance=int(distance),
                bridges=int(mask.sum()),
                mean_entropy_bits=float(entropies[mask].mean()),
            )
        )
    persistence = None
    if count:
        persistence = float(
            np.corrcoef(topics[:, :-1].ravel(), topics[:, 1:].ravel())[0, 1]
        )
        if not np.isfinite(persistence):
            persistence = None
    # Quantify filtering within individual contexts too; aggregate marginals can
    # hide large, opposing per-context shifts. Bound this extra diagnostic cost.
    context_tv, surviving_mass = [], []
    selected = np.unique(np.linspace(0, max(0, count - 1), min(count, 256)).astype(int))
    for index in selected:
        for position in range(1, config.n_cores):
            pre, post, dist = prior.probabilities(
                int(cores[index, position - 1]), topics[index, position]
            )
            allowed = (dist >= config.min_distance) & (dist <= config.max_distance)
            context_tv.append(float(np.abs(pre - post).sum() / 2))
            surviving_mass.append(float(pre[allowed].sum()))
    report = dict(
        examples=count,
        core_occurrences=int(core_counts.sum()),
        core_vocabulary_used=int((core_counts > 0).sum()),
        core_unigram_entropy_bits=_entropy(core_counts),
        core_frequencies_file="core_frequencies.tsv",
        consecutive_core_mi_bits=observed_mi,
        consecutive_core_pairs=int(len(pair_array)),
        distinct_consecutive_core_pairs=len(pairs),
        consecutive_core_mi_by_position_bits=[
            mutual_information(cores[:, j : j + 2]) for j in range(config.n_cores - 1)
        ],
        shuffled_mi_bits_mean=float(np.mean(null_mi)),
        shuffled_mi_bits_std=float(np.std(null_mi)),
        mi_excess_over_shuffle_bits=float(observed_mi - np.mean(null_mi)),
        mi_estimator="sparse plug-in; finite-sample upward bias; eight within-position shuffled controls",
        mi_interpretation="includes low-rank, topic, distance filtering, and baseline effects; not exclusively low-rank correlation",
        core_distances=_hist(distances.ravel()),
        connecting_path_entropy_bits=_hist(entropies.ravel()),
        sum_bridge_entropy_bits_per_document=_hist(entropies.sum(axis=1)),
        entropy_by_distance=by_distance,
        entropy_conditioning="exact RSP bridge entropy given each latent endpoint pair and bridge horizon; not entropy of the unmarked sequence",
        distance_filter_context_sample_count=len(context_tv),
        mean_context_filter_tv=float(np.mean(context_tv)) if context_tv else None,
        mean_pre_filter_mass_in_distance_band=float(np.mean(surviving_mass))
        if surviving_mass
        else None,
        intended_vs_realized={
            "baseline_to_realized_tv": float(np.abs(baseline - realized).sum() / 2),
            "soft_base_to_realized_tv": float(
                np.abs(prior.base_probabilities - realized).sum() / 2
            ),
            "pre_filter_to_post_filter_tv": float(np.abs(before - after).sum() / 2),
            "pre_filter_to_realized_tv": float(np.abs(before - realized).sum() / 2),
            "post_filter_to_realized_tv": float(np.abs(after - realized).sum() / 2),
            "expected_contexts": "realized core/topic histories; whole-chain retries and cross-split duplicate rejection may further bias frequencies",
        },
        baseline_expected_bridge_frequency=float(baseline @ prior.bridge_frequency),
        soft_base_expected_bridge_frequency=float(
            prior.base_probabilities @ prior.bridge_frequency
        ),
        realized_core_mean_pilot_bridge_frequency=float(
            realized @ prior.bridge_frequency
        ),
        core_hub_fraction=float(
            core_counts[graph.is_hub].sum() / max(1, core_counts.sum())
        ),
        connector_hub_fraction=float(
            connector_counts[graph.is_hub].sum() / max(1, connector_counts.sum())
        ),
        realized_topic_lag1_correlation=persistence,
        configured_topic_persistence=config.topic_persistence,
        frozen_factor_shapes=dict(
            source=list(prior.source_factors.shape),
            target=list(prior.target_factors.shape),
            topics=list(prior.topic_loadings.shape),
        ),
        pair_score_matrix_stored=False,
    )
    (out / "latent_core_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return report

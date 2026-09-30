"""Streaming exact n-gram counts, token statistics, and dependency-free SVG plots."""

from collections import Counter
import html
import json
from pathlib import Path
import sqlite3

import numpy as np

from .data import hints_from_row
from .graph import Graph, N_SPECIAL, PAD
from .rsp import RSP
from .routes import RouteDistribution, route_shortest


def _entropy(counts):
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]
    if not len(counts):
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _fit(counts):
    counts = np.sort(np.asarray(counts)[np.asarray(counts) > 0])[::-1]
    if len(counts) < 2:
        return None
    return float(
        np.polyfit(np.log(np.arange(1, len(counts) + 1)), np.log(counts), 1)[0]
    )


def concentration(counts):
    ranked = np.sort(np.asarray(counts)[np.asarray(counts) > 0])[::-1]
    total = int(ranked.sum())
    cumulative = ranked.cumsum() / max(1, total)
    return {
        "top_k_share": {
            str(k): float(ranked[:k].sum() / max(1, total))
            for k in (1, 5, 10, 20, 50, 100, 500, 1000)
        },
        "tokens_for_mass": {
            str(mass): int(np.searchsorted(cumulative, mass) + 1) if total else 0
            for mass in (0.25, 0.5, 0.75, 0.9)
        },
        "tokens_at_least_one_percent": int((ranked >= 0.01 * total).sum())
        if total
        else 0,
        "singletons": int((ranked == 1).sum()),
    }


def _plot(path, x, y, title, xlabel, ylabel, log=False):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if log:
        valid &= (x > 0) & (y > 0)
    x, y = x[valid], y[valid]
    if log:
        x, y = np.log10(x), np.log10(y)
        xlabel, ylabel = xlabel + " (log scale)", ylabel + " (log scale)"
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="440" viewBox="0 0 760 440">',
        '<rect width="760" height="440" fill="white"/>',
        '<path d="M70 40V370H730" fill="none" stroke="black"/>',
    ]
    if len(x):
        xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
        xx = 70 + 660 * (x - xmin) / max(xmax - xmin, 1e-9)
        yy = 370 - 310 * (y - ymin) / max(ymax - ymin, 1e-9)
        # Bound SVG size without changing statistics.
        for i in np.unique(np.linspace(0, len(x) - 1, min(len(x), 3000)).astype(int)):
            elements.append(
                f'<circle cx="{xx[i]:.2f}" cy="{yy[i]:.2f}" r="2" fill="#2463ac"/>'
            )
        xticks = (
            np.arange(np.ceil(xmin), np.floor(xmax) + 1)
            if log
            else np.linspace(xmin, xmax, 5)
        )
        yticks = (
            np.arange(np.ceil(ymin), np.floor(ymax) + 1)
            if log
            else np.linspace(ymin, ymax, 5)
        )
        for value in xticks:
            position = 70 + 660 * (value - xmin) / max(xmax - xmin, 1e-9)
            label = (
                (f"{10**value:,.0f}" if value >= 0 else f"{10**value:g}")
                if log
                else f"{value:.3g}"
            )
            elements.append(
                f'<text x="{position:.0f}" y="390" text-anchor="middle" font-size="11">{label}</text>'
            )
        for value in yticks:
            position = 374 - 310 * (value - ymin) / max(ymax - ymin, 1e-9)
            label = (
                (f"{10**value:,.0f}" if value >= 0 else f"{10**value:g}")
                if log
                else f"{value:.3g}"
            )
            elements.append(
                f'<text x="62" y="{position:.0f}" text-anchor="end" font-size="11">{label}</text>'
            )
    elements.extend(
        [
            f'<text x="380" y="24" text-anchor="middle">{html.escape(title)}</text>',
            f'<text x="400" y="425" text-anchor="middle">{html.escape(xlabel)}</text>',
            f'<text x="15" y="210" transform="rotate(-90 15 210)" text-anchor="middle">{html.escape(ylabel)}</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(elements))


def _hist(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": None, "counts": [], "bin_edges": []}
    counts, edges = np.histogram(values, bins=min(40, max(1, len(np.unique(values)))))
    return {
        "mean": float(values.mean()),
        "counts": counts.tolist(),
        "bin_edges": edges.tolist(),
    }


def diagnose(
    directory,
    output=None,
    split="train",
    ngrams=(1, 2, 3, 4, 5),
    max_examples=None,
    entropy_prompts=128,
    scope="path",
):
    root = Path(directory)
    out = Path(output) if output else root / f"diagnostics_{split}"
    out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata.get("task_mode") == "grammar_core":
        raise ValueError(
            "Grammar corpora use variable core counts; use the report.json produced "
            "by python -m tasks.hubpaths.grammar for grammar and n-gram diagnostics."
        )
    latent = metadata.get("task_mode") == "latent_core"
    if latent and scope == "prompt":
        raise ValueError(
            "latent-core data has no model-visible waypoint prompt; use core_frequencies.tsv"
        )
    graph = Graph.load(root / "graph.npz")
    c = graph.config
    prefix = 1 if latent else c.n_waypoints + 1
    tokens = np.load(root / f"{split}_tokens.npy", mmap_mode="r")
    masks = np.load(root / f"{split}_mask.npy", mmap_mode="r")
    metrics = np.load(root / f"{split}_metrics.npy", mmap_mode="r")
    count = min(len(tokens), max_examples) if max_examples is not None else len(tokens)
    orders = sorted(set(int(n) for n in ngrams))
    if any(n < 1 for n in orders):
        raise ValueError("n-gram orders must be positive")
    if scope not in {"path", "prompt", "full"}:
        raise ValueError("scope must be path, prompt, or full")
    dbpath = out / "ngram_counts.sqlite"
    if dbpath.exists():
        raise ValueError(f"diagnostics output already contains counts: {dbpath}")
    db = sqlite3.connect(dbpath)
    db.execute(
        "CREATE TABLE counts (n INTEGER, gram BLOB, count INTEGER NOT NULL, PRIMARY KEY(n,gram)) WITHOUT ROWID"
    )
    db.execute(
        "CREATE TABLE sequences (tokens BLOB PRIMARY KEY, count INTEGER NOT NULL) WITHOUT ROWID"
    )
    pending = Counter()
    frequency = np.zeros(graph.n_nodes + N_SPECIAL, dtype=np.int64)
    path_frequency = np.zeros(graph.n_nodes, dtype=np.int64)
    lengths, stretches, distances, hubs = [], [], [], []

    def flush():
        db.executemany(
            "INSERT INTO counts VALUES(?,?,?) ON CONFLICT(n,gram) DO UPDATE SET count=count+excluded.count",
            [(n, gram, int(v)) for (n, gram), v in pending.items()],
        )
        pending.clear()
        db.commit()

    for index in range(count):
        row = tokens[index]
        path = row[prefix:]
        path = path[path != PAD]
        nodes = path - N_SPECIAL
        np.add.at(path_frequency, nodes, 1)
        sequence = (
            path
            if scope == "path"
            else row[: prefix - 1]
            if scope == "prompt"
            else row[row != PAD]
        )
        np.add.at(frequency, sequence, 1)
        sequence = np.asarray(sequence, dtype="<i4")
        for n in orders:
            for pos in range(len(sequence) - n + 1):
                pending[(n, sequence[pos : pos + n].tobytes())] += 1
        db.execute(
            "INSERT INTO sequences VALUES(?,1) ON CONFLICT(tokens) DO UPDATE SET count=count+1",
            (sequence.tobytes(),),
        )
        lengths.append(len(path) - 1)
        distance = metrics[index, 2]
        distances.append(distance)
        stretches.append((len(path) - 1) / distance if distance else 1.0)
        hubs.append(int(graph.is_hub[nodes].sum()))
        if index % 512 == 511:
            flush()
    flush()
    ngram_stats = {}
    for n in orders:
        unique, total, singleton = db.execute(
            "SELECT COUNT(*),COALESCE(SUM(count),0),COALESCE(SUM(count=1),0) FROM counts WHERE n=?",
            (n,),
        ).fetchone()
        entropy = 0.0
        if total:
            for (v,) in db.execute("SELECT count FROM counts WHERE n=?", (n,)):
                probability = v / total
                entropy -= probability * np.log2(probability)
        top = [
            {"token_ids": np.frombuffer(gram, dtype="<i4").tolist(), "count": v}
            for gram, v in db.execute(
                "SELECT gram,count FROM counts WHERE n=? ORDER BY count DESC,gram LIMIT 20",
                (n,),
            )
        ]
        ngram_stats[str(n)] = {
            "total": total,
            "unique": unique,
            "singletons": singleton,
            "type_token_ratio": unique / total if total else 0.0,
            "repeat_fraction": 1 - unique / total if total else 0.0,
            "entropy_bits": float(entropy),
            "top": top,
        }
    unique_sequences = db.execute("SELECT COUNT(*) FROM sequences").fetchone()[0]
    db.close()
    entropy_values, shortest_counts = [], []
    rsp = RSP(graph, cache_dir=root / "goal_cache")
    sample_indices = np.unique(
        np.linspace(0, max(0, count - 1), min(count, entropy_prompts)).astype(int)
    )
    for index in sample_indices:
        row = tokens[index]
        if latent:
            entropy_values.append(float(metrics[index, 1]))
            continue
        waypoints = tuple(int(v) - N_SPECIAL for v in row[: c.n_waypoints])
        saved = metrics[index, 1]
        if np.isfinite(saved):
            entropy_values.append(float(saved))
        else:
            distribution = RouteDistribution(
                rsp, waypoints, hints_from_row(row, masks[index], c.n_waypoints)
            )
            entropy_values.append(distribution.cond_entropy_bits())
        shortest_counts.append(route_shortest(graph, waypoints)[1])
    degrees = np.diff(graph.adjacency.indptr)
    degree_rank_slope = _fit(degrees)
    degree_values, degree_counts = np.unique(degrees, return_counts=True)
    valid = degree_values > 0
    degree_pmf_slope = (
        float(
            np.polyfit(np.log(degree_values[valid]), np.log(degree_counts[valid]), 1)[0]
        )
        if valid.sum() > 1
        else None
    )
    h1 = _entropy(path_frequency)
    edge_counts = np.load(root / "train_edge_counts.npy")
    reverse = np.searchsorted(
        graph.edge_keys, graph.adjacency.indices * graph.n_nodes + graph.rows
    )
    covered = edge_counts + edge_counts[reverse]
    train_nodes = np.load(root / "train_node_counts.npy")
    report = {
        "split": split,
        "scope": scope,
        "examples_analyzed": count,
        "examples_available": len(tokens),
        "n_waypoints": metadata["latent_config"]["n_cores"]
        if latent
        else c.n_waypoints,
        "task_mode": "latent_core" if latent else "conditional_waypoint",
        "entropy_conditioning": metadata.get(
            "entropy_conditioning", "on model-visible waypoint prompt"
        ),
        "ngrams_cross_example_boundaries": False,
        "pad_included": False,
        "unique_sequences": unique_sequences,
        "duplicate_sequence_fraction": 1 - unique_sequences / count if count else 0.0,
        "token_count": int(frequency.sum()),
        "vocabulary_used": int((path_frequency > 0).sum()),
        "path_zipf_slope": _fit(path_frequency),
        "path_frequency_concentration": concentration(path_frequency),
        "path_unigram_entropy_bits": h1,
        "first_order_redundancy": 1 - h1 / np.log2(graph.n_nodes),
        "token_frequencies": [
            {
                "token_id": i,
                "node_id": i - N_SPECIAL if i >= N_SPECIAL else None,
                "count": int(v),
                "frequency": float(v / max(1, frequency.sum())),
            }
            for i, v in enumerate(frequency)
        ],
        "ngrams": ngram_stats,
        "path_length": _hist(lengths),
        "shortest_distance": _hist(distances),
        "stretch": _hist(stretches),
        "hubs_visited": _hist(hubs),
        "conditional_entropy_bits": _hist(entropy_values),
        "entropy_prompt_count": len(entropy_values),
        "exact_walk_statistics": not c.require_simple,
        "mean_distinct_shortest_paths": float(np.mean(shortest_counts))
        if shortest_counts
        else None,
        "degree": {
            "rank_slope": degree_rank_slope,
            "pmf_ols_exponent": -degree_pmf_slope
            if degree_pmf_slope is not None
            else None,
            "min": int(degrees.min()),
            "max": int(degrees.max()),
            "mean": float(degrees.mean()),
        },
        "train_coverage": {
            "node_fraction": float((train_nodes > 0).mean()),
            "edge_fraction": float((covered > 0).mean()),
            "min_undirected_edge_count": int(covered.min()),
            "edges_below_target": int((covered < c.min_edge_count).sum() // 2),
        },
    }
    if latent:
        from .latent_diagnostics import latent_report

        report["latent_core"] = latent_report(root, out, split, max_examples)
        report["exact_walk_statistics"] = False
        report["exact_bridge_statistics"] = True
        report["train_coverage"]["edges_below_target"] = None
    try:
        import powerlaw
    except ImportError:
        report["optional_powerlaw"] = None
    else:
        report["optional_powerlaw"] = {
            "degree_alpha": float(
                powerlaw.Fit(degrees, discrete=True, verbose=False).power_law.alpha
            ),
            "token_count_alpha": float(
                powerlaw.Fit(
                    path_frequency[path_frequency > 0], discrete=True, verbose=False
                ).power_law.alpha
            ),
        }
    (out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    with (out / "token_frequencies.tsv").open("w") as f:
        f.write("token_id\tnode_id\tcount\tfrequency\n")
        for row in report["token_frequencies"]:
            f.write(
                f"{row['token_id']}\t{row['node_id']}\t{row['count']}\t{row['frequency']:.12g}\n"
            )
    ranked = np.sort(path_frequency[path_frequency > 0])[::-1]
    _plot(
        out / "token_rank_frequency.svg",
        np.arange(1, len(ranked) + 1),
        ranked,
        "Path token rank-frequency",
        "rank",
        "count",
        True,
    )
    _plot(
        out / "degree_distribution.svg",
        degree_values,
        degree_counts,
        "Realized degree distribution",
        "degree",
        "nodes",
        True,
    )
    for key in (
        "path_length",
        "shortest_distance",
        "stretch",
        "hubs_visited",
        "conditional_entropy_bits",
    ):
        hist = report[key]
        centers = (np.asarray(hist["bin_edges"][:-1]) + hist["bin_edges"][1:]) / 2
        _plot(
            out / f"{key}.svg",
            centers,
            hist["counts"],
            key.replace("_", " ").title(),
            "value",
            "count",
        )
    return report

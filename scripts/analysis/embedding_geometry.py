"""Geometry of a trained diffusion embedding table (RePlaid paper Sec. 5.1, App. T).

CPU only. Three stages, each writing into ``--out``:

``corpus``  (project env) reference rows from the task's held-out validation split.
            Rebuilds the TinyGSM split (length filter, then ``train_test_split``
            with the config's ``val_ratio``/``val_seed``) from token lengths alone,
            so it does not write the 12M-row tokenized cache.
``pos``     (an env with spaCy and transformers; no torch) dominant Universal POS
            per token: decode, tag with spaCy, re-encode with the fast tokenizer's
            offsets, inherit the tag of the maximally overlapping word, tally a
            V x 18 count matrix (17 UPOS + SPACE), argmax.
``analyze`` (project env) loads each checkpoint through the runner's config
            migration and ``load_weights`` (EMA and/or raw), then PCA scree on the
            row-normalized table, t-SNE coloured by POS and token category,
            k-means / Ward / HDBSCAN clusterings scored by silhouette and by
            agreement with POS (NMI, ARI, purity), kNN POS accuracy, nearest
            neighbours, and cross-checkpoint similarity. Figures (PNG), CSVs and
            ``summary.json`` hold every number.

    uv run --locked python scripts/analysis/embedding_geometry.py corpus \\
        --config BUNDLE/config.yaml --rows 20000 --out OUT
    SPACY_PY scripts/analysis/embedding_geometry.py pos --out OUT --n-process 16
    uv run --locked python scripts/analysis/embedding_geometry.py analyze \\
        --checkpoint replaid=BUNDLE_A --checkpoint hybrid=BUNDLE_B --out OUT
"""

import argparse
import json
import keyword
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
UPOS = (
    "ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X SPACE"
).split()
CATEGORIES = (
    "special", "whitespace", "digit", "operator", "punctuation", "keyword", "identifier", "word",
)
PYTHON_WORDS = set(keyword.kwlist) | {
    "print", "range", "len", "int", "float", "str", "round", "sum", "min", "max", "abs",
    "list", "dict", "append", "self", "result", "solution", "math", "import",
}


# --------------------------------------------------------------------- corpus
def _load_task_tokenizer(config_path):
    sys.path.insert(0, str(ROOT))
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    config = OmegaConf.load(config_path)
    task = instantiate(config.task)
    return config, task, task.tokenizer


def corpus(args):
    import pyarrow.parquet as pq

    config, task, tokenizer = _load_task_tokenizer(args.config)
    data = config.task.data
    if data.get("data_path") or data.get("max_examples") is not None or data.wrap:
        raise ValueError("only the full TinyGSM hub split (no data_path/max_examples/wrap) is rebuilt")
    files = sorted(Path(args.parquet_dir).glob("*.parquet"))
    if not files:
        raise ValueError(f"no parquet files in {args.parquet_dir}")
    started = time.time()
    separator = tokenizer(data.separator, add_special_tokens=False)["input_ids"]
    keep, offset = [], 0
    for file in files:
        table = pq.read_table(file, columns=["question", "code"])
        questions = [q.strip() for q in table.column("question").to_pylist()]
        codes = [c.strip() for c in table.column("code").to_pylist()]
        for start in range(0, len(questions), 200_000):
            q = tokenizer(questions[start:start + 200_000], add_special_tokens=False)["input_ids"]
            a = tokenizer(codes[start:start + 200_000], add_special_tokens=False)["input_ids"]
            lengths = np.fromiter((len(x) + len(y) for x, y in zip(q, a)), dtype=np.int64)
            lengths += 2 + len(separator)
            if data.filter_too_long:
                keep.append(offset + start + np.flatnonzero(lengths <= task.length))
            else:
                keep.append(offset + start + np.arange(len(lengths)))
        offset += len(questions)
        print(f"{file.name}: {offset} rows, {time.time() - started:.0f}s", flush=True)
    kept = np.concatenate(keep)
    # datasets.Dataset.train_test_split(test_size, seed): ceil test count, rng permutation.
    n_test = math.ceil(data.val_ratio * len(kept))
    permutation = np.random.default_rng(data.val_seed).permutation(len(kept))
    validation = kept[permutation[:n_test]]
    chosen = validation[: args.rows]
    rows = []
    by_file = np.cumsum([0] + [pq.ParquetFile(f).metadata.num_rows for f in files])
    for index in sorted(chosen.tolist()):
        f = int(np.searchsorted(by_file, index, side="right") - 1)
        rows.append((index, f, index - by_file[f]))
    tables = {}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokens = 0
    with open(out / "corpus.jsonl", "w") as handle:
        for index, f, local in rows:
            if f not in tables:
                tables[f] = pq.read_table(files[f], columns=["question", "code"])
            question = tables[f].column("question")[int(local)].as_py().strip()
            code = tables[f].column("code")[int(local)].as_py().strip()
            q = tokenizer(question, add_special_tokens=False)["input_ids"]
            a = tokenizer(code, add_special_tokens=False)["input_ids"]
            tokens += len(q) + len(separator) + len(a)
            handle.write(json.dumps({"index": index, "prompt_ids": q + separator, "answer_ids": a}) + "\n")
    summary = {
        "source": [str(f) for f in files],
        "tokenizer": data.tokenizer_name_or_path,
        "vocab_size": len(tokenizer),
        "special_ids": {"bos": tokenizer.bos_token_id, "eos": tokenizer.eos_token_id,
                        "pad": tokenizer.pad_token_id},
        "raw_rows": int(offset),
        "rows_after_length_filter": int(len(kept)),
        "validation_rows": int(n_test),
        "val_ratio": data.val_ratio,
        "val_seed": data.val_seed,
        "rows_used": len(rows),
        "content_tokens_used": int(tokens),
        "selection": "first rows of the validation permutation (a uniform random subset)",
        "seconds": time.time() - started,
    }
    (out / "corpus.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


# ------------------------------------------------------------------------ pos
def pos(args):
    import spacy
    from transformers import AutoTokenizer

    out = Path(args.out)
    meta = json.loads((out / "corpus.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(meta["tokenizer"])
    vocab = meta["vocab_size"]
    rows = [json.loads(line) for line in open(out / "corpus.jsonl")]
    texts, boundaries = [], []
    for row in rows:
        text = tokenizer.decode(row["prompt_ids"] + row["answer_ids"])
        prefix = tokenizer.decode(row["prompt_ids"])
        texts.append(text)
        boundaries.append(len(prefix) if text.startswith(prefix) else -1)
    nlp = spacy.load(args.model, disable=["parser", "ner", "lemmatizer"])
    nlp.max_length = 10**6
    index = {tag: i for i, tag in enumerate(UPOS)}
    counts = np.zeros((vocab, len(UPOS)), dtype=np.int64)
    regions = np.zeros((vocab, 2), dtype=np.int64)  # occurrences in prompt, answer
    started, words = time.time(), 0
    docs = nlp.pipe(texts, n_process=args.n_process, batch_size=args.batch_size)
    for row, (text, boundary, doc) in enumerate(zip(texts, boundaries, docs)):
        owner = np.full(len(text), -1, dtype=np.int64)
        tags = np.empty(len(doc), dtype=np.int64)
        for j, word in enumerate(doc):
            if not word.is_space:
                owner[word.idx: word.idx + len(word.text)] = j
            tags[j] = index.get(word.pos_, index["X"])
        words += len(doc)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        for token, (start, end) in zip(encoded["input_ids"], encoded["offset_mapping"]):
            covered = owner[start:end]
            covered = covered[covered >= 0]
            tag = index["SPACE"] if len(covered) == 0 else tags[np.bincount(covered).argmax()]
            counts[token, tag] += 1
            regions[token, int(boundary >= 0 and start >= boundary)] += 1
        if row % 2000 == 0:
            print(f"{row}/{len(rows)} rows, {time.time() - started:.0f}s", flush=True)
    np.savez_compressed(out / "pos_counts.npz", counts=counts, regions=regions, tags=np.array(UPOS))
    summary = {
        "spacy_model": args.model,
        "spacy_version": spacy.__version__,
        "rows": len(rows),
        "spacy_words": int(words),
        "reencoded_tokens": int(counts.sum()),
        "distinct_tokens_observed": int((counts.sum(1) > 0).sum()),
        "rows_without_prefix_boundary": int(sum(b < 0 for b in boundaries)),
        "seconds": time.time() - started,
    }
    (out / "pos.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


# -------------------------------------------------------------------- analyze
def token_category(text, special, code_fraction):
    stripped = text.strip()
    if special:
        return "special"
    if not stripped:
        return "whitespace"
    if stripped.isdigit() or stripped.replace(".", "", 1).isdigit():
        return "digit"
    if all(not c.isalnum() and c != "_" for c in stripped):
        operators = set("+-*/=<>%^&|~!")
        return "operator" if any(c in operators for c in stripped) else "punctuation"
    if stripped in PYTHON_WORDS:
        return "keyword"
    if "_" in stripped or (code_fraction >= 0.9 and stripped.isascii()):
        return "identifier"
    return "word"


def load_tables(bundle, use_ema):
    """The diffusion table (and the hybrid's clean-context map) via the codebase's loaders."""
    sys.path.insert(0, str(ROOT))
    import torch
    from omegaconf import OmegaConf
    from core.models.revealed_context import RevealedContextInput
    from core.representations.embedding import EmbeddingRepresentation
    from core.training.checkpoints import load_weights
    from core.training.legacy_checkpoints import migrate_legacy_config
    from core.training.module import TrainingModule

    config = migrate_legacy_config(OmegaConf.load(Path(bundle) / "config.yaml"))
    config.compile = False
    module = TrainingModule(config)
    if not use_ema:
        module.ema = None
    step = load_weights(module, str(Path(bundle) / "model.ckpt"))
    tables = {}
    with torch.no_grad(), module.evaluation_weights(use_ema):
        seen = set()
        for name, component in module.algorithm.named_modules():
            if isinstance(component, EmbeddingRepresentation) and hasattr(component, "table"):
                if id(component.table) not in seen:
                    seen.add(id(component.table))
                    tables.setdefault("E", (name, component.table.detach().double().numpy().copy()))
            if isinstance(component, RevealedContextInput):
                weight = component.context_projection.weight.detach().double().numpy().copy()
                tables["context_projection"] = (name, weight, float(component.context_scale))
    if "E" not in tables:
        raise ValueError("no trainable embedding table in this checkpoint")
    return step, config, tables


def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)


def scree(x):
    from sklearn.decomposition import PCA

    ratio = PCA().fit(x).explained_variance_ratio_
    cumulative = np.cumsum(ratio)
    return {
        "explained_variance_ratio": ratio.tolist(),
        "cumulative": cumulative.tolist(),
        "pcs_90": int(np.searchsorted(cumulative, 0.90) + 1),
        "pcs_99": int(np.searchsorted(cumulative, 0.99) + 1),
        "participation_ratio": float(ratio.sum() ** 2 / (ratio**2).sum()),
    }


def purity(labels, truth):
    total = 0
    for cluster in np.unique(labels):
        total += np.bincount(truth[labels == cluster]).max()
    return total / len(truth)


def agreement(labels, truth):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    return {
        "nmi": float(normalized_mutual_info_score(truth, labels)),
        "ari": float(adjusted_rand_score(truth, labels)),
        "purity": float(purity(labels, truth)),
    }


def knn_accuracy(x, truth, k=10):
    from sklearn.neighbors import NearestNeighbors

    neighbours = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(x)
    _, index = neighbours.kneighbors(x)
    votes = truth[index[:, 1:]]
    predicted = np.array([np.bincount(v, minlength=truth.max() + 1).argmax() for v in votes])
    return float((predicted == truth).mean())


def linear_cka(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    cross = np.linalg.norm(a.T @ b) ** 2
    return float(cross / (np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)))


def analyze_table(x, labels, args, rng_seed, with_clusters=True):
    """Everything about one (row-normalized) table restricted to observed tokens."""
    from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
    from sklearn.metrics import silhouette_score

    result = {"knn10_pos_accuracy": knn_accuracy(x, labels["pos"])}
    result["knn10_category_accuracy"] = knn_accuracy(x, labels["category"])
    if not with_clusters:
        return result, {}
    clusters, rows = {}, []
    sample = np.random.default_rng(rng_seed).choice(len(x), min(len(x), 10_000), replace=False)
    for k in args.k:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=rng_seed).fit(x)
        ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(x)
        for method, assign in (("kmeans", kmeans.labels_), ("ward", ward.labels_)):
            row = {
                "method": method,
                "k": k,
                "silhouette_cosine": float(silhouette_score(x[sample], assign[sample], metric="cosine")),
                **{f"pos_{m}": v for m, v in agreement(assign, labels["pos"]).items()},
                **{f"category_{m}": v for m, v in agreement(assign, labels["category"]).items()},
            }
            rows.append(row)
            clusters[(method, k)] = assign
    density = HDBSCAN(min_cluster_size=args.min_cluster_size).fit(x)
    assigned = density.labels_ >= 0
    hdb = {
        "method": "hdbscan",
        "k": int(density.labels_.max() + 1),
        "min_cluster_size": args.min_cluster_size,
        "noise_fraction": float(1 - assigned.mean()),
        "silhouette_cosine": float(silhouette_score(x[assigned], density.labels_[assigned], metric="cosine"))
        if density.labels_.max() > 0 else None,
        **{f"pos_{m}": v for m, v in agreement(density.labels_[assigned], labels["pos"][assigned]).items()},
        **{f"category_{m}": v for m, v in agreement(density.labels_[assigned], labels["category"][assigned]).items()},
    }
    rows.append(hdb)
    clusters[("hdbscan", hdb["k"])] = density.labels_
    result["clustering"] = rows
    kmeans_rows = [r for r in rows if r["method"] == "kmeans"]
    best = max(kmeans_rows, key=lambda r: r["silhouette_cosine"])
    result["kmeans_best_silhouette"] = best
    result["kmeans_best_pos_nmi"] = max(kmeans_rows, key=lambda r: r["pos_nmi"])
    return result, clusters


def plot_scree(path, curves, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, axes = plt.subplots(1, len(curves), figsize=(4.6 * len(curves), 3.4), squeeze=False)
    for axis, (name, s), colour in zip(axes[0], curves.items(), colours):
        ratio = np.array(s["explained_variance_ratio"])
        pcs = np.arange(1, len(ratio) + 1)
        axis.bar(pcs, ratio, color=colour, width=0.7)
        axis.set_xlabel("Principal component")
        axis.set_ylabel("Explained variance ratio")
        twin = axis.twinx()  # cumulative share on [0, 1] (same quantity, cumulated)
        twin.plot(pcs, s["cumulative"], color="#52514e", linewidth=2, marker="o", markersize=3)
        twin.set_ylim(0, 1.02)
        twin.set_ylabel("Cumulative explained variance")
        for level, key in ((0.9, "pcs_90"), (0.99, "pcs_99")):
            twin.axhline(level, color="#9b9a95", linewidth=0.8, linestyle=":")
        axis.set_title(f"{name}\n90%: PC {s['pcs_90']}   99%: PC {s['pcs_99']}", fontsize=9)
        axis.spines[["top"]].set_visible(False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_tsne(path, xy, labels, names, title, top=7):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    frequency = Counter(labels.tolist())
    shown = [label for label, _ in frequency.most_common(top)]
    fig, axis = plt.subplots(figsize=(7.2, 6.4))
    other = ~np.isin(labels, shown)
    if other.any():
        axis.scatter(xy[other, 0], xy[other, 1], s=2, color="#c3c2b7", alpha=0.6,
                     label=f"other ({other.sum()})", linewidths=0)
    for label, colour in zip(shown, palette):
        mask = labels == label
        axis.scatter(xy[mask, 0], xy[mask, 1], s=2.5, color=colour, alpha=0.8,
                     label=f"{names[label]} ({mask.sum()})", linewidths=0)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=10)
    axis.legend(markerscale=5, fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_cluster_metrics(path, per_model):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    metrics = ("silhouette_cosine", "pos_nmi", "pos_ari", "pos_purity")
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.3))
    for axis, metric in zip(axes, metrics):
        for (name, rows), colour in zip(per_model.items(), colours):
            for method, style in (("kmeans", "-"), ("ward", "--")):
                selected = [r for r in rows if r["method"] == method]
                axis.plot([r["k"] for r in selected], [r[metric] for r in selected], style,
                          color=colour, linewidth=2, marker="o", markersize=3,
                          label=f"{name} {method}")
        axis.set_xscale("log", base=2)
        axis.set_xlabel("k")
        axis.set_title(metric, fontsize=9)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def analyze(args):
    from sklearn.manifold import TSNE
    from sklearn.metrics import normalized_mutual_info_score
    from transformers import AutoTokenizer

    out = Path(args.out)
    corpus_meta = json.loads((out / "corpus.json").read_text())
    pos_meta = json.loads((out / "pos.json").read_text())
    saved = np.load(out / "pos_counts.npz")
    counts, regions = saved["counts"], saved["regions"]
    tokenizer = AutoTokenizer.from_pretrained(corpus_meta["tokenizer"])
    vocab = counts.shape[0]
    total = counts.sum(1)
    observed = np.flatnonzero(total >= args.min_count)
    dominant = counts.argmax(1)
    special = set(corpus_meta["special_ids"].values())
    strings = [tokenizer.decode([v]) for v in range(vocab)]
    code_fraction = regions[:, 1] / regions.sum(1).clip(1)
    category = np.array([
        CATEGORIES.index(token_category(strings[v], v in special, code_fraction[v])) for v in range(vocab)
    ])
    labels = {"pos": dominant[observed], "category": category[observed]}
    p = counts[observed] / total[observed, None]
    entropy = -(np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0)).sum(1)
    summary = {
        "corpus": corpus_meta,
        "pos": pos_meta,
        "observed_tokens": int(len(observed)),
        "min_count": args.min_count,
        "pos_distribution": {UPOS[t]: int(c) for t, c in Counter(labels["pos"].tolist()).most_common()},
        "category_distribution": {CATEGORIES[t]: int(c) for t, c in Counter(labels["category"].tolist()).most_common()},
        "mean_pos_entropy_bits": float(entropy.mean()),
        "pos_vs_category_nmi": float(normalized_mutual_info_score(labels["pos"], labels["category"])),
        "checkpoints": {},
    }
    frequent = observed[np.argsort(-total[observed])]
    probes = [int(v) for v in frequent[:12]]
    for text in (" 5", "5", " +", " =", " *", "def", " return", " total", " apples",
                 " she", " each", " cost", " dollars", "(", " if", " per"):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) == 1 and ids[0] in set(observed.tolist()) and ids[0] not in probes:
            probes.append(ids[0])
    token_rows = {
        "token_id": observed,
        "token": [strings[v] for v in observed],
        "count": total[observed],
        "answer_fraction": code_fraction[observed],
        "dominant_pos": [UPOS[t] for t in labels["pos"]],
        "pos_entropy_bits": entropy,
        "category": [CATEGORIES[c] for c in labels["category"]],
    }
    normalized_tables = {}
    curves = {}
    cluster_rows = {}
    for spec in args.checkpoint:
        name, bundle = spec.split("=", 1)
        for weights in args.weights:
            tag = f"{name}/{weights}"
            print(f"loading {tag}", flush=True)
            step, config, tables = load_tables(bundle, weights == "ema")
            table_name, table = tables["E"]
            raw_norms = np.linalg.norm(table, axis=1)
            unit = normalize(table)
            normalized_tables[tag] = unit
            entry = {
                "bundle": bundle,
                "checkpoint_step": step,
                "weights": weights,
                "table": table_name,
                "shape": list(table.shape),
                "row_norm_mean": float(raw_norms.mean()),
                "row_norm_std": float(raw_norms.std()),
                "scree_all_rows": scree(unit),
                "scree_observed_rows": scree(unit[observed]),
                "scree_unobserved_rows": scree(unit[np.setdiff1d(np.arange(vocab), observed)])
                if len(observed) < vocab - 16 else None,
            }
            curves[f"{tag} all {vocab}"] = entry["scree_all_rows"]
            curves[f"{tag} observed {len(observed)}"] = entry["scree_observed_rows"]
            # Unobserved rows: how far from a random isotropic table are they?
            mean_direction = unit.mean(0)
            entry["mean_row_norm_of_average"] = float(np.linalg.norm(mean_direction))
            if "context_projection" in tables:
                projection_name, weight, scale = tables["context_projection"]
                context = scale * table @ weight.T
                entry["revealed_context"] = {
                    "module": projection_name,
                    "shape": list(context.shape),
                    "note": "linear image of E (no separate prompt table); rank <= state_dim",
                    "projection_singular_values": np.linalg.svd(weight, compute_uv=False).tolist(),
                    "scree_observed_rows": scree(normalize(context[observed])),
                    "knn10_overlap_with_E": None,
                }
                from sklearn.neighbors import NearestNeighbors

                a = NearestNeighbors(n_neighbors=11, metric="cosine").fit(unit[observed]).kneighbors()[1]
                cn = normalize(context[observed])
                b = NearestNeighbors(n_neighbors=11, metric="cosine").fit(cn).kneighbors()[1]
                entry["revealed_context"]["knn10_overlap_with_E"] = float(np.mean([
                    len(set(x[:10]) & set(y[:10])) / 10 for x, y in zip(a, b)
                ]))
                entry["revealed_context"]["knn10_pos_accuracy"] = knn_accuracy(cn, labels["pos"])
            heavy = weights == args.weights[0]
            result, clusters = analyze_table(unit[observed], labels, args, args.seed, with_clusters=heavy)
            entry.update(result)
            if heavy:
                cluster_rows[tag] = result["clustering"]
                print(f"t-SNE {tag}", flush=True)
                xy = TSNE(n_components=2, perplexity=args.perplexity, init="pca",
                          random_state=args.seed, metric="cosine").fit_transform(unit[observed])
                safe = tag.replace("/", "_")
                token_rows[f"{safe}_tsne_x"], token_rows[f"{safe}_tsne_y"] = xy[:, 0], xy[:, 1]
                plot_tsne(out / f"tsne_pos_{safe}.png", xy, labels["pos"], UPOS,
                          f"{name} ({weights}): t-SNE of E rows, dominant POS ({len(observed)} tokens)")
                plot_tsne(out / f"tsne_category_{safe}.png", xy, labels["category"], CATEGORIES,
                          f"{name} ({weights}): t-SNE of E rows, token category", top=8)
                best_k = result["kmeans_best_silhouette"]["k"]
                described = {}
                for k in sorted({best_k, 18}):
                    assign = clusters[("kmeans", k)]
                    token_rows[f"{safe}_kmeans{k}"] = assign
                    groups = []
                    for c in np.argsort(-np.bincount(assign))[: args.describe]:
                        members = observed[assign == c]
                        top_tokens = members[np.argsort(-total[members])][:15]
                        groups.append({
                            "cluster": int(c),
                            "size": int(len(members)),
                            "top_tokens": [strings[v] for v in top_tokens],
                            "pos": {UPOS[t]: int(n) for t, n in Counter(dominant[members].tolist()).most_common(3)},
                            "category": {CATEGORIES[t]: int(n) for t, n in Counter(category[members].tolist()).most_common(3)},
                        })
                    described[str(k)] = groups
                entry["kmeans_clusters"] = described
                similarity = unit[observed] @ unit[probes].T
                entry["nearest_neighbours"] = {
                    strings[v]: [strings[observed[j]] for j in np.argsort(-similarity[:, i])[1:9]]
                    for i, v in enumerate(probes)
                }
                entry["nearest_neighbour_cosines"] = {
                    strings[v]: [round(float(similarity[j, i]), 3) for j in np.argsort(-similarity[:, i])[1:9]]
                    for i, v in enumerate(probes)
                }
            summary["checkpoints"][tag] = entry
    names = list(normalized_tables)
    comparisons = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x, y = normalized_tables[a][observed], normalized_tables[b][observed]
            from sklearn.neighbors import NearestNeighbors

            nx = NearestNeighbors(n_neighbors=11, metric="cosine").fit(x).kneighbors()[1]
            ny = NearestNeighbors(n_neighbors=11, metric="cosine").fit(y).kneighbors()[1]
            comparisons[f"{a} vs {b}"] = {
                "linear_cka_observed": linear_cka(x, y),
                "knn10_overlap_observed": float(np.mean([
                    len(set(p[:10]) & set(q[:10])) / 10 for p, q in zip(nx, ny)
                ])),
                "mean_row_cosine": float((normalized_tables[a] * normalized_tables[b]).sum(1).mean()),
            }
    summary["comparisons"] = comparisons
    plot_scree(out / "scree_observed.png", {k: v for k, v in curves.items() if "observed" in k},
               "PCA scree of row-normalized E (tokens observed in the reference corpus)")
    plot_scree(out / "scree_all.png", {k: v for k, v in curves.items() if "observed" not in k},
               "PCA scree of row-normalized E (all vocabulary rows)")
    plot_cluster_metrics(out / "cluster_metrics.png", cluster_rows)
    _write_csvs(out, summary, curves, cluster_rows, token_rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({tag: {
        "pcs_90_99_all": (e["scree_all_rows"]["pcs_90"], e["scree_all_rows"]["pcs_99"]),
        "pcs_90_99_observed": (e["scree_observed_rows"]["pcs_90"], e["scree_observed_rows"]["pcs_99"]),
        "knn10_pos_accuracy": e["knn10_pos_accuracy"],
    } for tag, e in summary["checkpoints"].items()}, indent=2))


def _write_csvs(out, summary, curves, cluster_rows, token_rows):
    import csv

    with open(out / "scree.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["table", "pc", "explained_variance_ratio", "cumulative"])
        for name, s in curves.items():
            for i, (r, c) in enumerate(zip(s["explained_variance_ratio"], s["cumulative"]), 1):
                writer.writerow([name, i, r, c])
    with open(out / "clustering.csv", "w", newline="") as handle:
        fields = sorted({key for rows in cluster_rows.values() for row in rows for key in row})
        writer = csv.DictWriter(handle, ["checkpoint"] + fields)
        writer.writeheader()
        for name, rows in cluster_rows.items():
            for row in rows:
                writer.writerow({"checkpoint": name, **row})
    with open(out / "tokens.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(token_rows))
        for values in zip(*token_rows.values()):
            writer.writerow([json.dumps(v) if isinstance(v, str) else v for v in values])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("corpus")
    c.add_argument("--config", required=True, help="a checkpoint bundle's config.yaml")
    c.add_argument("--parquet-dir", required=True, help="TinyGSM/TinyGSM data/ parquet shards")
    c.add_argument("--rows", type=int, default=20000)
    c.add_argument("--out", required=True)
    p = commands.add_parser("pos")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="en_core_web_sm")
    p.add_argument("--n-process", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    a = commands.add_parser("analyze")
    a.add_argument("--checkpoint", action="append", required=True, help="NAME=BUNDLE_DIR")
    a.add_argument("--out", required=True)
    a.add_argument("--weights", nargs="+", default=["ema", "raw"], choices=["ema", "raw"])
    a.add_argument("--k", type=int, nargs="+", default=[2, 4, 8, 12, 18, 24, 32, 48, 64])
    a.add_argument("--min-count", type=int, default=1)
    a.add_argument("--min-cluster-size", type=int, default=15)
    a.add_argument("--perplexity", type=float, default=30.0)
    a.add_argument("--describe", type=int, default=8)
    a.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    {"corpus": corpus, "pos": pos, "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    main()

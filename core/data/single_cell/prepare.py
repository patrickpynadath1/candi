"""Prepare CFGen AnnData splits without densifying the entire count matrix."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from . import DATASETS, SOURCE_URL


def count_block(matrix, rows):
    block = matrix[rows]
    block = block.toarray() if sparse.issparse(block) else np.asarray(block)
    if (
        not np.isfinite(block).all()
        or (block < 0).any()
        or not np.equal(block, np.floor(block)).all()
    ):
        raise ValueError(
            "Expected nonnegative integer counts; select the raw-count layer explicitly"
        )
    if (block > np.iinfo(np.int32).max - 1).any():
        raise ValueError("Counts exceed int32 storage")
    return block.astype(np.int32)


def prepare(
    dataset,
    train_path,
    test_path,
    output,
    *,
    layer="X_counts",
    label_key=None,
    count_cap=None,
    validation_fraction=0.1,
    seed=42,
    chunk_size=128,
):
    """Fit vocabulary on training cells, reserving one tail/overflow bin.

    Default retains all training counts exactly. An explicit count_cap collapses
    counts >= cap to the cap token. Nothing is fit to validation or test cells.
    """
    import anndata as ad

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if not 0 < validation_fraction < 1 or chunk_size < 1:
        raise ValueError("Require validation_fraction in (0,1) and positive chunk_size")
    if count_cap is not None and (not isinstance(count_cap, int) or count_cap < 1):
        raise ValueError("count_cap must be a positive integer")
    train_path, test_path, output = Path(train_path), Path(test_path), Path(output)
    if train_path.resolve() == test_path.resolve():
        raise ValueError("Training and test paths must differ")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared dataset: {output}")
    train, test = ad.read_h5ad(train_path), ad.read_h5ad(test_path)
    if not train.var_names.is_unique or not test.var_names.is_unique:
        raise ValueError("Gene identifiers must be unique")
    if set(train.var_names) != set(test.var_names):
        raise ValueError("Train/test gene sets differ")
    test = test[:, train.var_names].copy()
    if not train.obs_names.is_unique or not test.obs_names.is_unique:
        raise ValueError("Cell identifiers must be unique within each source split")
    if set(train.obs_names) & set(test.obs_names):
        raise ValueError("Train/test cell identifiers overlap; verify source splits")
    label_key = label_key or DATASETS[dataset]["label_key"]
    for data in (train, test):
        if label_key not in data.obs or data.obs[label_key].isna().any():
            raise ValueError(f"Missing cell labels in obs[{label_key!r}]")
        if layer != "X" and layer not in data.layers:
            raise ValueError(
                f"Missing count layer {layer!r}; no fallback to potentially normalized X"
            )
    matrices = [
        data.X if layer == "X" else data.layers[layer] for data in (train, test)
    ]
    rng = np.random.default_rng(seed)
    order = rng.permutation(train.n_obs)
    n_valid = max(1, round(train.n_obs * validation_fraction))
    if n_valid >= train.n_obs or test.n_obs == 0:
        raise ValueError("Train, validation and test splits must all be nonempty")
    split_rows = {
        "train": order[n_valid:],
        "validation": order[:n_valid],
        "test": rng.permutation(test.n_obs),
    }
    categories = sorted(set(train.obs.iloc[split_rows["train"]][label_key].astype(str)))
    label_to_id = {label: i for i, label in enumerate(categories)}
    all_labels = []
    for data in (train, test):
        labels = data.obs[label_key].astype(str)
        unknown = set(labels) - set(categories)
        if unknown:
            raise ValueError(
                f"Labels absent from training split: {sorted(unknown)}; choose another split seed"
            )
        all_labels.append(np.array([label_to_id[x] for x in labels], dtype=np.int32))
    if count_cap is None:
        maximum = 0
        rows = split_rows["train"]
        for start in range(0, len(rows), chunk_size):
            maximum = max(
                maximum,
                int(count_block(matrices[0], rows[start : start + chunk_size]).max()),
            )
        count_cap = maximum + 1
        encoding = "raw_counts_with_training_overflow"
    else:
        encoding = "capped_counts"
    if count_cap + 1 < len(categories):
        raise ValueError(
            "Count vocabulary must also accommodate the cell-type prefix labels"
        )
    output.mkdir(parents=True)
    split_info = {}
    for split, rows in split_rows.items():
        source = int(split == "test")
        tokens = np.lib.format.open_memmap(
            output / f"{split}.npy",
            mode="w+",
            dtype=np.int32,
            shape=(len(rows), train.n_vars),
        )
        saturated = 0
        for start in range(0, len(rows), chunk_size):
            block = count_block(matrices[source], rows[start : start + chunk_size])
            saturated += int((block >= count_cap).sum())
            tokens[start : start + len(block)] = np.minimum(block, count_cap)
        tokens.flush()
        np.save(output / f"{split}_labels.npy", all_labels[source][rows])
        np.save(output / f"{split}_rows.npy", rows)
        split_info[split] = {
            "cells": len(rows),
            "tail_fraction": saturated / tokens.size,
        }
    genes = train.var_names.tolist()
    (output / "genes.json").write_text(json.dumps(genes))
    metadata = {
        "format_version": 1,
        "dataset": dataset,
        "source_url": SOURCE_URL,
        "source_paths": {
            "train": str(train_path.resolve()),
            "test": str(test_path.resolve()),
        },
        "source_sizes": {
            "train": train_path.stat().st_size,
            "test": test_path.stat().st_size,
        },
        "layer": layer,
        "label_key": label_key,
        "labels": categories,
        "num_genes": len(genes),
        "vocab_size": count_cap + 1,
        "count_cap": count_cap,
        "encoding": encoding,
        "data_seed": seed,
        "validation_fraction": validation_fraction,
        "gene_order_sha256": hashlib.sha256(json.dumps(genes).encode()).hexdigest(),
        "splits": split_info,
    }
    # Written last: a partial preparation is never a valid cache.
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata

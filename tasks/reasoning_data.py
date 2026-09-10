"""Reproducible conversion of public reasoning releases to memory-mapped Arrow."""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

VERSION = 1
ENCODING_VERSION = {"zebra": 2}
EQ_R = {
    "sudoku_extreme": "sudoku-extreme-1k-aug-1000",
    "maze_unique": "maze-30x30-unique-1k",
}
DRIVE = {
    "zebra-test-data.pkl": "1LCzeK8p5Z48AlOVn5BtPkj2Tqgvui7yH",
    "zebra-train-data.pkl": "1UHoalXvhDtSYm3D3Dgm6YQgcTQKT7Ahf",
    "Sudoku-test-data.npy": "1QMnUMkGcO_E_loAYFDmNo27dA3aryVcx",
    "Sudoku-train-data.npy": "1u9jw_MykbfbGSRNXjXQtmzT5I_geWHLN",
}


class ArrayUnpickler(pickle.Unpickler):
    """Only NumPy arrays/scalars and builtin pickle containers are accepted."""

    def find_class(self, module, name):
        if (module, name) in {
            ("numpy", "ndarray"),
            ("numpy", "dtype"),
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy.core.multiarray", "scalar"),
        }:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"unsupported pickle global: {module}.{name}")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_files(raw, task):
    raw = Path(raw)
    if task in EQ_R:
        return sorted((raw / "eqr" / EQ_R[task]).glob("*/*.npy"))
    prefix, suffix = ("Sudoku", "npy") if task == "sudoku_shah" else ("zebra", "pkl")
    return [raw / "shah" / f"{prefix}-{s}-data.{suffix}" for s in ("train", "test")]


def source_records(raw, task, split, layout=None, limit=None):
    """Yield canonical records; source answers/traces never enter the prompt."""
    raw = Path(raw)
    count = 0
    if task in EQ_R:
        base = raw / "eqr" / EQ_R[task] / split
        inputs = np.load(base / "all__inputs.npy", mmap_mode="r")
        labels = np.load(base / "all__labels.npy", mmap_mode="r")
        groups = np.load(base / "all__group_indices.npy", mmap_mode="r")
        indices = np.load(base / "all__puzzle_indices.npy", mmap_mode="r")
        selected_rows = (
            np.linspace(0, len(inputs) - 1, min(limit, len(inputs)), dtype=int)
            if limit
            else range(len(inputs))
        )
        for i in selected_rows:
            group = int(np.searchsorted(indices[groups], i, side="right") - 1)
            side = 9 if task == "sudoku_extreme" else 30
            offset = 1 if task == "sudoku_extreme" else 0
            puzzle = (inputs[i].reshape(side, side) - offset).tolist()
            answer = (labels[i].reshape(side, side) - offset).tolist()
            yield dict(
                task=task,
                puzzle=puzzle,
                answer=answer,
                source_id=f"{task}/{split}/{i}",
                group_id=f"{task}/{split}/base_{group}",
            )
            count += 1
            if limit and count >= limit:
                return
    elif task == "sudoku_shah":
        data = np.load(raw / "shah" / f"Sudoku-{split}-data.npy", mmap_mode="r")
        for i, row in enumerate(data):
            n = int(row[0])
            cells = row[1:].reshape(81, 4)
            if not 0 <= n <= 81 or len(set(map(tuple, cells[:, :2]))) != 81:
                raise ValueError(f"invalid Sudoku coordinates at {split}/{i}")
            if (
                np.any(cells[:, :2] < 0)
                or np.any(cells[:, :2] > 8)
                or np.any(cells[:, 2] < 1)
                or np.any(cells[:, 2] > 9)
            ):
                raise ValueError(f"invalid Sudoku values at {split}/{i}")
            if np.any(cells[:n, 3] != 0):
                raise ValueError("invalid given-cell marker")
            answer = np.zeros((9, 9), dtype=int)
            puzzle = answer.copy()
            answer[cells[:, 0], cells[:, 1]] = cells[:, 2]
            puzzle[cells[:n, 0], cells[:n, 1]] = cells[:n, 2]
            yield dict(
                task=task,
                puzzle=puzzle.tolist(),
                answer=answer.tolist(),
                source_id=f"{task}/{split}/{i}",
                group_id=digest(puzzle.tolist()),
                clue_count=n,
            )
            count += 1
            if limit and count >= limit:
                return
    else:
        from .zebra import parse_clues

        with (raw / "shah" / f"zebra-{split}-data.pkl").open("rb") as stream:
            data = ArrayUnpickler(stream).load()
        for i, (sequence, table, ordering) in enumerate(data):
            table = np.asarray(table)
            if not np.array_equal(table[0], np.arange(table.shape[1])):
                raise ValueError("unexpected house ordering")
            a, h = table.shape[0] - 1, table.shape[1]
            if layout and (a, h) != tuple(layout):
                continue
            clues, tokens = parse_clues(sequence)
            canonical = sorted(json.dumps(c, sort_keys=True) for c in clues)
            yield dict(
                task=task,
                puzzle=tokens,
                answer=table[1:].astype(int).tolist(),
                source_id=f"{task}/{split}/{i}",
                group_id=digest([a, h, canonical]),
                attributes=a,
                houses=h,
            )
            count += 1
            if limit and count >= limit:
                return


def is_development(group, seed, fraction):
    return int(digest([seed, group])[:16], 16) / 2**64 < fraction


def prepare(
    raw,
    output,
    task,
    layout=None,
    seed=42,
    dev_fraction=0.05,
    limit=None,
    full_training=False,
    audit_labels=True,
):
    from datasets import Dataset, DatasetDict, Features, Sequence, Value
    from datasets.arrow_writer import ArrowWriter
    from .reasoning import get_reasoning_tokenizer, encode_record, score_record

    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be in (0, 1)")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"prepared datasets are immutable: {output}")
    tokenizer = get_reasoning_tokenizer(task)
    prompt, length = (
        (91, 180)
        if task.startswith("sudoku_")
        else (902, 1802)
        if task == "maze_unique"
        else (476, 512)
    )
    files = source_files(raw, task)
    if not files or any(not p.exists() for p in files):
        raise FileNotFoundError(f"missing raw files for {task}")
    sources = []
    for p in files:
        entry = dict(
            path=str(p.relative_to(raw)), bytes=p.stat().st_size, sha256=file_hash(p)
        )
        if task in EQ_R:
            relative = p.relative_to(Path(raw) / "eqr")
            meta = (
                Path(raw)
                / "eqr/.cache/huggingface/download"
                / str(relative.parent)
                / (relative.name + ".metadata")
            )
            revision = meta.read_text().splitlines()[0] if meta.exists() else None
            entry.update(
                repository="locuslab/EqR-data",
                revision=revision,
                url=f"https://huggingface.co/datasets/locuslab/EqR-data/resolve/{revision or 'main'}/{relative}",
            )
        else:
            entry["url"] = (
                f"https://drive.usercontent.google.com/download?id={DRIVE[p.name]}&export=download"
            )
        sources.append(entry)
    manifest = dict(
        version=VERSION,
        encoding_version=ENCODING_VERSION.get(task, 1),
        task=task,
        layout=layout,
        seed=seed,
        dev_fraction=dev_fraction,
        source_limit=limit,
        full_training=full_training,
        sources=sources,
        prompt_length=prompt,
        length=length,
        vocab_size=len(tokenizer),
        vocabulary=list(getattr(tokenizer, "symbols", map(str, range(len(tokenizer))))),
        profile="source_subset_smoke"
        if limit
        else "full_training_fixed_budget"
        if full_training
        else "group_heldout_development",
    )
    temp = output.with_name(output.name + ".preparing")
    temp.mkdir(parents=True, exist_ok=False)
    features = Features(
        {
            **{
                k: Sequence(Value("uint8"), length=length)
                for k in ("input_ids", "attention_mask", "prompt_mask", "padding_mask")
            },
            **{k: Value("string") for k in ("source_id", "group_id", "metadata")},
        }
    )
    writers = {
        s: ArrowWriter(path=str(temp / f"{s}.arrow"), features=features)
        for s in ("train", "validation", "test")
    }
    # Preserve the released test set and remove any identical symbolic puzzle
    # from the training source. A puzzle group is determined by layout plus its
    # canonical clue set, so clue-order permutations cannot cross the boundary.
    heldout_groups = (
        {
            record["group_id"]
            for record in source_records(raw, task, "test", layout, limit)
        }
        if task == "zebra"
        else set()
    )
    excluded_train_test_groups = set()
    excluded_train_test_records = 0
    counts = dict.fromkeys(writers, 0)
    groups = {s: set() for s in writers}
    puzzle_hashes = {s: set() for s in writers}
    gold_checks = dict.fromkeys(writers, 0)
    with (temp / "selected_ids.jsonl").open("w") as selected:
        for source_split in ("train", "test"):
            for record in source_records(raw, task, source_split, layout, limit):
                if source_split == "train" and record["group_id"] in heldout_groups:
                    excluded_train_test_groups.add(record["group_id"])
                    excluded_train_test_records += 1
                    continue
                dev = is_development(record["group_id"], seed, dev_fraction)
                split = (
                    "test"
                    if source_split == "test"
                    else "validation"
                    if dev and not full_training
                    else "train"
                )
                encoded = encode_record(record, tokenizer, prompt, length)
                # Full audit is intentional: invalid source labels must fail preparation.
                if audit_labels:
                    prediction = np.asarray(encoded["input_ids"])[
                        np.asarray(encoded["attention_mask"], dtype=bool)
                    ]
                    score = score_record(prediction.tolist(), record, tokenizer)
                    good = score.get(
                        "optimal_path",
                        score.get(
                            "constraints_satisfied",
                            score.get(
                                "valid_solution", score.get("valid_solution_rate", True)
                            ),
                        ),
                    )
                    if not score["exact"] or not good:
                        raise ValueError(
                            f"invalid gold label: {record['source_id']}: {score}"
                        )
                    gold_checks[split] += 1
                writers[split].write(encoded)
                counts[split] += 1
                groups[split].add(record["group_id"])
                puzzle_hashes[split].add(digest(record["puzzle"]))
                selected.write(
                    json.dumps(
                        dict(
                            split=split,
                            source_id=record["source_id"],
                            group_id=record["group_id"],
                        )
                    )
                    + "\n"
                )
    for writer in writers.values():
        writer.finalize()
    if (
        not counts["train"]
        or not counts["test"]
        or (not full_training and not counts["validation"])
    ):
        raise ValueError(
            f"empty required split: {counts}; increase --limit or change development seed"
        )
    if groups["train"] & groups["validation"]:
        raise AssertionError("augmentation group leakage")
    data = DatasetDict(
        {s: Dataset.from_file(str(temp / f"{s}.arrow")) for s in writers if counts[s]}
    )
    data.save_to_disk(str(temp / "dataset"))
    for s in writers:
        (temp / f"{s}.arrow").unlink()
    manifest.update(
        counts=counts,
        decontamination=dict(
            train_against_test=task == "zebra",
            excluded_train_records=excluded_train_test_records,
            excluded_train_groups=len(excluded_train_test_groups),
        ),
        group_counts={s: len(v) for s, v in groups.items()},
        gold_checks=gold_checks,
        exact_puzzle_overlap={
            f"{a}:{b}": len(puzzle_hashes[a] & puzzle_hashes[b])
            for a, b in [
                ("train", "validation"),
                ("train", "test"),
                ("validation", "test"),
            ]
        },
    )
    manifest["fingerprint"] = digest(manifest)
    (temp / "manifest.json").write_text(json.dumps(manifest, indent=2))
    temp.rename(output)
    return manifest

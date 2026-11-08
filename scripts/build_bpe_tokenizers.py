"""Train nested byte-BPE vocabularies on parquet text and audit full-story lengths.

Train the largest vocabulary once; smaller vocabularies keep a prefix of the
same learned merge sequence. No validation text participates in BPE training.
"""

import argparse
import csv
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL = ["<pad>", "<bos>", "<eos>", "<mask>"]
BASE_SIZE = 256 + len(SPECIAL)


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def batches(files):
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=1024, columns=["text"]
        ):
            yield [text.strip() for text in batch.column(0).to_pylist()]


def prefix_tokenizer(tokenizer, size):
    """Keep the initial alphabet and the first learned merges, with stable IDs."""
    state = json.loads(tokenizer.to_str())
    size = min(size, tokenizer.get_vocab_size())
    state["model"]["vocab"] = {
        token: index for token, index in state["model"]["vocab"].items() if index < size
    }
    state["model"]["merges"] = state["model"]["merges"][: size - BASE_SIZE]
    vocab = state["model"]["vocab"]
    for a, b in state["model"]["merges"]:
        assert a in vocab and b in vocab and a + b in vocab
    return Tokenizer.from_str(json.dumps(state))


def summarize(lengths):
    values = np.asarray(lengths, dtype=np.int32)
    return {
        "stories": len(values),
        "mean_tokens": float(values.mean()),
        "median_tokens": float(np.median(values)),
        "p90_tokens": float(np.quantile(values, 0.9)),
        "p95_tokens": float(np.quantile(values, 0.95)),
        "p99_tokens": float(np.quantile(values, 0.99)),
        "max_tokens": int(values.max()),
        "mean_with_bos_eos": float(values.mean() + 2),
        **{
            f"fit_{n}_with_bos_eos_pct": float(100 * np.mean(values + 2 <= n))
            for n in (256, 512, 1024)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--vocab-sizes", type=int, nargs="+", default=[260, 500, 1000, 5000, 10000]
    )
    args = parser.parse_args()
    sizes = sorted(set(args.vocab_sizes))
    if min(sizes) < BASE_SIZE:
        parser.error(f"Lossless byte BPE requires at least {BASE_SIZE} tokens")
    if (args.output / "manifest.json").exists() or any(args.output.glob("bpe-*.json")):
        parser.error(
            "Output already contains an audit/tokenizers; reuse it or choose a new output directory"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    files = {
        split: sorted(args.data.glob(f"{split}-*.parquet"))
        for split in ("train", "validation")
    }
    if any(not paths for paths in files.values()):
        parser.error("Both train and validation parquet shards are required")
    manifest = {
        "status": "running",
        "versions": {name: version(name) for name in ("tokenizers", "pyarrow")},
        "source": str(args.data.resolve()),
        "vocab_sizes": sizes,
        "special_tokens": SPECIAL,
        "preprocessing": "strip only; preserve case and Unicode",
        "training_split": "train",
        "method": "byte BPE; nested prefixes of one learned merge sequence",
        "sequence_length_convention": "content tokens; BOS/EOS reported separately; no truncation or filtering",
        "files": {
            split: [
                {
                    "name": p.name,
                    "rows": pq.ParquetFile(p).metadata.num_rows,
                    "sha256": checksum(p),
                }
                for p in paths
            ]
            for split, paths in files.items()
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    started = time.monotonic()
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=max(sizes),
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=SPECIAL,
        show_progress=False,
    )
    print("Training BPE on full train split", flush=True)
    tokenizer.train_from_iterator(
        batches(files["train"]),
        trainer=trainer,
        length=sum(pq.ParquetFile(p).metadata.num_rows for p in files["train"]),
    )
    print(
        f"BPE trained in {time.monotonic() - started:.1f}s; vocab={tokenizer.get_vocab_size()}",
        flush=True,
    )
    variants = {size: prefix_tokenizer(tokenizer, size) for size in sizes}
    for size, tok in variants.items():
        tok.save(str(args.output / f"bpe-{size}.json"))
    rows = []
    # Validation first gives quick held-out numbers, then measure the entire train set.
    for split in ("validation", "train"):
        lengths = {size: [] for size in sizes}
        for batch_index, texts in enumerate(batches(files[split])):
            for size, tok in variants.items():
                if size == BASE_SIZE:
                    lengths[size].extend(len(text.encode("utf-8")) for text in texts)
                else:
                    encoded = tok.encode_batch(texts, add_special_tokens=False)
                    lengths[size].extend(len(x.ids) for x in encoded)
                    if split == "validation" or batch_index == 0:
                        assert (
                            tok.decode_batch(
                                [x.ids for x in encoded], skip_special_tokens=False
                            )
                            == texts
                        )
            if batch_index % 100 == 0:
                print(f"{split}: {len(lengths[sizes[0]])} stories measured", flush=True)
        for size in sizes:
            row = {
                "split": split,
                "requested_vocab": size,
                "actual_vocab": variants[size].get_vocab_size(),
                **summarize(lengths[size]),
            }
            rows.append(row)
            np.save(
                args.output / f"lengths-{split}-{size}.npy",
                np.asarray(lengths[size], dtype=np.int32),
            )
            print(json.dumps(row), flush=True)
        (args.output / "summary.json").write_text(json.dumps(rows, indent=2))
        with (args.output / "summary.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    manifest["status"] = "complete"
    manifest["validation_roundtrip_verified"] = True
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Complete in {time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()

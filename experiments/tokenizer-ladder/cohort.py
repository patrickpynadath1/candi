"""Build the tokenizer ladder's shared story cohort (data.allowed_ids_path).

    python experiments/tokenizer-ladder/cohort.py PARQUET_DIR [--output FILE]

Every vocabulary trains and evaluates on the same stories: a story is kept only if
its prompt and continuation fit the sequence capacity of all five tokenizers, and
validation stories that also occur in the training split are dropped. The committed
evaluation indices (evaluation/*.json) index this cohort's validation split, so the
file must be rebuilt exactly; the script prints its sha256 next to the published one.

PARQUET_DIR holds the TinyStories parquet shards (``train-*.parquet``,
``validation-*.parquet``) at revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.data.tinystories import StoryBPETokenizer  # noqa: E402

# Sequence capacity per vocabulary; the same values as bpe.capacity in
# configs/sweep/tokenizer_ladder.yaml.
CAPACITY = {260: 1024, 500: 768, 1000: 512, 5000: 512, 10000: 512}
PROMPT_BYTES = 64
TOKENIZERS = "artifacts/tokenizers/tinystories-bpe/bpe-{}.json"
PUBLISHED_SHA256 = "f16da32a1308b16dc13b2558b7eaf22fdd5bdfefa875d0147c4c2cba7aad12c8"
# Where configs/sweep/tokenizer_ladder.yaml looks for it (data.allowed_ids_path).
DEFAULT_OUTPUT = (
    Path(os.environ.get("CANDI_STORAGE_ROOT", "~/.cache/candi"))
    / "data_cache/tinystories_bpe/tokenizer-ladder-cohort.json"
)


def split_story(text, prompt_bytes=PROMPT_BYTES):
    """(prompt, continuation, sha256) exactly as core.data.tinystories.prepare_stories."""
    raw = text.strip().encode("utf-8")
    if len(raw) <= prompt_bytes:
        return None
    cut = prompt_bytes
    while cut and (raw[cut] & 0xC0) == 0x80:
        cut -= 1
    if cut == 0:
        return None
    return raw[:cut].decode(), raw[cut:].decode(), hashlib.sha256(raw).hexdigest()


def stories(parquet_dir, split):
    """Unique complete stories of one split, first occurrence kept."""
    import pyarrow.parquet as pq

    rows, seen = [], set()
    files = sorted(Path(parquet_dir).glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {split}-*.parquet shards in {parquet_dir}")
    for file in files:
        for batch in pq.ParquetFile(file).iter_batches(10000, columns=["text"]):
            for text in batch.column(0).to_pylist():
                parts = split_story(text)
                if parts and parts[2] not in seen:
                    seen.add(parts[2])
                    rows.append(parts)
    return rows


def fitting(rows, vocab, capacity):
    """Story hashes whose BOS + prompt + continuation + EOS fit ``capacity`` tokens."""
    tokenizer = StoryBPETokenizer(TOKENIZERS.format(vocab)).tokenizer
    keep = set()
    for start in range(0, len(rows), 20000):
        chunk = rows[start : start + 20000]
        prompts = tokenizer.encode_batch([r[0] for r in chunk], add_special_tokens=False)
        continuations = tokenizer.encode_batch([r[1] for r in chunk], add_special_tokens=False)
        for parts, a, b in zip(chunk, prompts, continuations):
            if len(a.ids) + len(b.ids) + 2 <= capacity:
                keep.add(parts[2])
    return keep


def build(parquet_dir, capacity=CAPACITY):
    """Sorted ``split:sha256`` ids of the cohort, and a count report."""
    ids, report = {}, {}
    for split in ("train", "validation"):
        rows = stories(parquet_dir, split)
        report[f"{split}_unique"] = len(rows)
        cohort = {r[2] for r in rows}
        for vocab, length in capacity.items():
            keep = fitting(rows, vocab, length)
            report[f"{split}_fit_{vocab}"] = len(keep)
            cohort &= keep
        ids[split] = cohort
    overlap = ids["train"] & ids["validation"]
    ids["validation"] -= overlap
    report.update(
        validation_removed_as_train_duplicates=len(overlap),
        train_cohort=len(ids["train"]),
        validation_cohort=len(ids["validation"]),
    )
    return sorted(f"{split}:{key}" for split, keys in ids.items() for key in keys), report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parquet_dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    values, report = build(args.parquet_dir)
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Same serialization as the published file, so the hashes are comparable.
    output.write_text(json.dumps(values))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps(report, indent=1))
    print(f"wrote {output}\nsha256 {digest}")
    print("matches the published cohort" if digest == PUBLISHED_SHA256 else f"differs from the published cohort {PUBLISHED_SHA256}")
    return output


if __name__ == "__main__":
    main()

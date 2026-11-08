"""Rebuild the two data inputs of the published runs that are too large to commit.

    # text8: the Hugging Face afmck/text8 copy, written as the task's split files.
    python scripts/prepare_data.py text8 \
        --parquet PATH/afmck-text8/data --raw-dir $CANDI_STORAGE_ROOT/data_cache/text8/raw
    # Only check split files you already have (e.g. derived from text8.zip):
    python scripts/prepare_data.py text8 \
        --raw-dir $CANDI_STORAGE_ROOT/data_cache/text8/raw

    # TinyStories: the story cohort that the evaluation indices refer to.
    python scripts/prepare_data.py tinystories \
        --parquet PATH/roneneldan-TinyStories/data \
        --output $CANDI_STORAGE_ROOT/data_cache/tinystories_bpe/allowed_ids.json

Both commands compare what they produce with the SHA-256 of the files the published
runs used and exit non-zero on a mismatch. Download the parquet files yourself, at the
revisions below (``huggingface-cli download --repo-type dataset NAME --revision REV``).
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT8_REVISION = "58c74e966ccc66eab2de6b52fad5b14ddb259fa4"  # afmck/text8
TEXT8_SHA256 = {
    "train": "af83e65b7d7aa95dfe2c571da8f3e2d1b0c385f177140a21f3db07520a3ac6f4",
    "validation": "c9395f11fb185c7cf73f57134be37b6a32c6364ce36e9cec3127d27c7b8bc5dd",
    "test": "b336f67a7616a0d94ed6cc2049437928aafe01318e76a84abe6918320660f98c",
}
TINYSTORIES_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"  # roneneldan/TinyStories
COHORT_SHA256 = "f16da32a1308b16dc13b2558b7eaf22fdd5bdfefa875d0147c4c2cba7aad12c8"
# The cohort keeps stories that fit every tokenizer of the ladder at its capacity,
# so all vocabulary sizes see the same stories.
CAPACITY = {260: 1024, 500: 768, 1000: 512, 5000: 512, 10000: 512}
PROMPT_BYTES = 64
TOKENIZERS = ROOT / "artifacts/tokenizers/tinystories-bpe"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text8(parquet, raw_dir):
    """text8.{train,validation,test}.txt from the parquet copy (one row per split)."""
    import pyarrow.parquet as pq

    raw_dir.mkdir(parents=True, exist_ok=True)
    for split in TEXT8_SHA256:
        (file,) = Path(parquet).glob(f"{split}-*.parquet")
        (text,) = pq.read_table(file).column(0).to_pylist()
        (raw_dir / f"text8.{split}.txt").write_text(text)


def check_text8(raw_dir):
    """Splits whose hash differs from the published runs' files."""
    return [split for split, digest in TEXT8_SHA256.items() if sha256(raw_dir / f"text8.{split}.txt") != digest]


def split_story(text):
    """(prompt, continuation, id) as core.data.tinystories prepares a story; None if too short."""
    raw = text.strip().encode("utf-8")
    if len(raw) <= PROMPT_BYTES:
        return None
    cut = PROMPT_BYTES
    while cut and (raw[cut] & 0xC0) == 0x80:
        cut -= 1
    if cut == 0:
        return None
    return raw[:cut].decode(), raw[cut:].decode(), hashlib.sha256(raw).hexdigest()


def stories(parquet, split):
    import pyarrow.parquet as pq

    rows, seen = [], set()
    for file in sorted(Path(parquet).glob(f"{split}-*.parquet")):
        for batch in pq.ParquetFile(file).iter_batches(10000, columns=["text"]):
            for text in batch.column(0).to_pylist():
                parts = split_story(text)
                if parts and parts[2] not in seen:
                    seen.add(parts[2])
                    rows.append(parts)
    return rows


def fitting(rows, tokenizer_path, capacity):
    """Ids of stories whose prompt + continuation + 2 special tokens fit ``capacity``."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    keep = set()
    for start in range(0, len(rows), 20000):
        chunk = rows[start : start + 20000]
        prompts = tokenizer.encode_batch([r[0] for r in chunk], add_special_tokens=False)
        continuations = tokenizer.encode_batch([r[1] for r in chunk], add_special_tokens=False)
        for row, a, b in zip(chunk, prompts, continuations):
            if len(a.ids) + len(b.ids) + 2 <= capacity:
                keep.add(row[2])
    return keep


def cohort(parquet, tokenizers=TOKENIZERS):
    """Sorted ``split:id`` strings; validation stories also in train are dropped."""
    ids = {}
    for split in ("train", "validation"):
        rows = stories(parquet, split)
        keep = {r[2] for r in rows}
        for vocabulary, capacity in CAPACITY.items():
            keep &= fitting(rows, Path(tokenizers) / f"bpe-{vocabulary}.json", capacity)
        ids[split] = keep
    ids["validation"] -= ids["train"]
    return sorted(f"{split}:{key}" for split, keys in ids.items() for key in keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    text8 = commands.add_parser("text8", help="write and/or check the text8 split files")
    text8.add_argument("--parquet", type=Path, help=f"afmck/text8 data/ directory (revision {TEXT8_REVISION})")
    text8.add_argument("--raw-dir", type=Path, required=True, help="the task's data.raw_dir")
    tiny = commands.add_parser("tinystories", help="rebuild the TinyStories story allowlist")
    tiny.add_argument("--parquet", type=Path, required=True, help=f"TinyStories data/ directory (revision {TINYSTORIES_REVISION})")
    tiny.add_argument("--output", type=Path, required=True, help="where data.allowed_ids_path points")
    args = parser.parse_args()
    if args.command == "text8":
        if args.parquet:
            write_text8(args.parquet, args.raw_dir)
        different = check_text8(args.raw_dir)
        if different:
            raise SystemExit(f"text8 {different} differ from the published runs' splits")
        print(f"text8 splits in {args.raw_dir} match the published runs")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(cohort(args.parquet)))
        if sha256(args.output) != COHORT_SHA256:
            raise SystemExit(f"{args.output} differs from the published runs' cohort")
        print(f"{args.output} matches the published runs' cohort")


if __name__ == "__main__":
    main()

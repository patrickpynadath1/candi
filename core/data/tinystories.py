"""Bounded, cached complete-story data with an offline byte vocabulary."""

import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile

from torch.utils.data import Dataset



def split_story(text, prompt_bytes):
    """(prompt, continuation, id) of a complete story, or None if it is too short.

    The prompt is the first ``prompt_bytes`` bytes, backed up to a UTF-8 character
    boundary; the id is the SHA-256 of the whole stripped story. The published
    cohort (scripts/prepare_data.py) depends on this split staying the same.
    """
    raw = text.strip().encode("utf-8")
    # Complete stories only: never convert a truncation into a fake ending.
    if len(raw) <= prompt_bytes:
        return None
    cut = prompt_bytes
    while cut and (raw[cut] & 0xC0) == 0x80:
        cut -= 1
    if cut == 0:
        return None
    return raw[:cut].decode("utf-8"), raw[cut:].decode("utf-8"), hashlib.sha256(raw).hexdigest()

class StoryByteTokenizer:
    """UTF-8 bytes plus PAD/BOS/EOS/MASK; no external tokenizer download."""

    pad_token_id, bos_token_id, eos_token_id, mask_token_id = range(4)
    vocab_size = 260

    def __len__(self):
        return self.vocab_size

    def encode(self, text):
        return [value + 4 for value in text.encode("utf-8")]

    def decode(self, ids, skip_special_tokens=True):
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def decode_bytes(self, ids):
        return bytes(int(i) - 4 for i in ids if 4 <= int(i) < 260)


class StoryBPETokenizer:
    """Load a saved tokenizer without changing TinyStories' special-token contract."""

    pad_token_id, bos_token_id, eos_token_id, mask_token_id = range(4)

    def __init__(self, path):
        from tokenizers import Tokenizer

        path = Path(path)
        if not path.is_absolute() and not path.exists():
            # Repository-relative (e.g. artifacts/tokenizers/...), from any working dir.
            path = Path(__file__).resolve().parents[2] / path
        content = path.read_bytes()
        self.cache_identity = hashlib.sha256(content).hexdigest()
        self.tokenizer = Tokenizer.from_str(content.decode("utf-8"))
        state = json.loads(content)
        if state["model"]["type"] != "BPE" or state["decoder"]["type"] != "ByteLevel":
            raise ValueError("Story BPE requires a byte-level BPE tokenizer")
        for index, token in enumerate(("<pad>", "<bos>", "<eos>", "<mask>")):
            if self.tokenizer.token_to_id(token) != index:
                raise ValueError("Story BPE requires PAD/BOS/EOS/MASK at IDs 0/1/2/3")
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.vocab_size = self.tokenizer.get_vocab_size()
        # Invert ByteLevel's reversible byte-to-Unicode alphabet. Work in bytes
        # before UTF-8 decoding so invalid generations remain measurable.
        visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        remaining = [value for value in range(256) if value not in visible]
        self.byte_decoder = dict(
            zip(
                map(chr, visible + list(range(256, 256 + len(remaining)))),
                visible + remaining,
            )
        )

    def __len__(self):
        return self.vocab_size

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids, skip_special_tokens=True):
        return self.tokenizer.decode(
            list(map(int, ids)), skip_special_tokens=skip_special_tokens
        )

    def decode_bytes(self, ids):
        pieces = (
            self.tokenizer.id_to_token(int(i))
            for i in ids
            if 4 <= int(i) < self.vocab_size
        )
        return bytes(self.byte_decoder[char] for piece in pieces for char in piece)


class StoryDataset(Dataset):
    def __init__(self, records, manifest=None):
        self.records = records
        self.manifest = manifest or {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def _source(data, split):
    path = data.get(f"{split}_path")
    limit = int(data[f"max_{split}_examples"])
    if limit < 1:
        raise ValueError("TinyStories source limits must be positive")
    if data.get("parquet_dir"):
        import pyarrow.parquet as pq

        files = sorted(Path(data.parquet_dir).glob(f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No {split} parquet shards in {data.parquet_dir}")
        remaining = limit
        for file in files:
            for batch in pq.ParquetFile(file).iter_batches(
                batch_size=1024, columns=["text"]
            ):
                texts = batch.column(0).to_pylist()[:remaining]
                yield from ({"text": text} for text in texts)
                remaining -= len(texts)
                if remaining == 0:
                    return
    elif path:
        with Path(path).open() as handle:
            for line in itertools.islice(handle, limit):
                yield json.loads(line)
    else:
        if not data.get("download", False):
            raise FileNotFoundError(
                f"Provide data.{split}_path JSONL or set data.download=true "
                "to prepare a bounded TinyStories cache"
            )
        from datasets import load_dataset

        source = load_dataset(
            data.dataset_name,
            split=split,
            revision=data.get("revision"),
            streaming=True,
            cache_dir=str(Path(data.cache_dir) / "hub"),
        )
        yield from itertools.islice(source, limit)


def prepare_stories(data, split, length, prompt_bytes, tokenizer=None):
    if split not in {"train", "validation"}:
        raise ValueError(f"unknown TinyStories split: {split}")
    local = data.get(f"{split}_path")
    identity = {
        "version": 1,
        "dataset": data.dataset_name,
        "revision": data.get("revision"),
        "split": split,
        "limit": int(data[f"max_{split}_examples"]),
        "length": length,
        "prompt_bytes": prompt_bytes,
        "source_hash": hashlib.sha256(Path(local).read_bytes()).hexdigest()
        if local
        else None,
    }
    if data.get("parquet_dir"):
        digest = hashlib.sha256()
        files = sorted(Path(data.parquet_dir).glob(f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No {split} parquet shards in {data.parquet_dir}")
        for file in files:
            digest.update(file.name.encode())
            with file.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        identity["source_hash"] = digest.hexdigest()
    allowed = None
    if data.get("allowed_ids_path"):
        content = Path(data.allowed_ids_path).read_bytes()
        identity["allowed_ids_hash"] = hashlib.sha256(content).hexdigest()
        allowed = set(json.loads(content))
    tokenizer = tokenizer if tokenizer is not None else StoryByteTokenizer()
    if hasattr(tokenizer, "cache_identity"):
        # Preserve the identity of existing byte caches; BPE gets its own cache.
        identity["tokenizer"] = tokenizer.cache_identity
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(data.cache_dir)
    cache = root / f"stories_{split}_{digest[:24]}.json"
    if cache.exists():
        payload = json.loads(cache.read_text())
        return StoryDataset(payload["records"], payload.get("manifest"))
    records, seen = [], set()
    considered = 0
    for example in _source(data, split):
        considered += 1
        parts = split_story(example["text"], prompt_bytes)
        if parts is None:
            continue
        prompt, reference, key = parts
        # Encode separately so a merge can never straddle the prompt boundary.
        prompt_ids, reference_ids = (
            tokenizer.encode(prompt),
            tokenizer.encode(reference),
        )
        ids = (
            [tokenizer.bos_token_id]
            + prompt_ids
            + reference_ids
            + [tokenizer.eos_token_id]
        )
        if len(ids) > length:
            continue
        if allowed is not None and f"{split}:{key}" not in allowed:
            continue
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "id": f"{split}:{key}",
                "input_ids": ids,
                "prompt_length": len(prompt_ids) + 1,
                "prompt": prompt,
                "reference": reference,
            }
        )
    if not records:
        raise ValueError(
            "No complete TinyStories fit the length/prompt limits; increase task.length or source limit"
        )
    manifest = {
        "source_examples": considered,
        "retained_examples": len(records),
        "retention_rate": len(records) / considered,
        "cache_identity": digest,
    }
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=root, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(
            {"identity": identity, "manifest": manifest, "records": records}, handle
        )
    os.replace(temporary, cache)
    return StoryDataset(records, manifest)

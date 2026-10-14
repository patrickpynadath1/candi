"""Bounded, cached complete-story data with an offline byte vocabulary."""

import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile

from torch.utils.data import Dataset


class StoryByteTokenizer:
    """UTF-8 bytes plus PAD/BOS/EOS/MASK; no external tokenizer download."""

    pad_token_id, bos_token_id, eos_token_id, mask_token_id = range(4)
    vocab_size = 260

    def __len__(self):
        return self.vocab_size

    def encode(self, text):
        return [value + 4 for value in text.encode("utf-8")]

    def decode(self, ids, skip_special_tokens=True):
        return bytes(int(i) - 4 for i in ids if 4 <= int(i) < 260).decode(
            "utf-8", errors="replace"
        )


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
    if path:
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


def prepare_stories(data, split, length, prompt_bytes):
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
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(data.cache_dir)
    cache = root / f"stories_{split}_{digest[:24]}.json"
    if cache.exists():
        payload = json.loads(cache.read_text())
        return StoryDataset(payload["records"], payload.get("manifest"))
    tokenizer = StoryByteTokenizer()
    records, seen = [], set()
    considered = 0
    for example in _source(data, split):
        considered += 1
        text = example["text"].strip()
        raw = text.encode("utf-8")
        # Complete stories only: never convert a truncation into a fake ending.
        if not prompt_bytes < len(raw) <= length - 2:
            continue
        # Back up to a valid UTF-8 boundary, never split a multibyte character.
        cut = prompt_bytes
        while cut and (raw[cut] & 0xC0) == 0x80:
            cut -= 1
        if cut == 0:
            continue
        key = hashlib.sha256(raw).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "id": f"{split}:{key}",
                "input_ids": [1] + tokenizer.encode(text) + [2],
                "prompt_length": cut + 1,
                "prompt": raw[:cut].decode("utf-8"),
                "reference": raw[cut:].decode("utf-8"),
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

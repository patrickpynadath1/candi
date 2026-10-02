"""Character Text8 encoding, raw splits, and deterministic legacy-style packing."""

from pathlib import Path
import hashlib
import json
import os
import re
import tempfile
import urllib.request
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset


class Text8Tokenizer:
    symbols = (
        "[CLS]",
        "[SEP]",
        "[BOS]",
        "[EOS]",
        "[MASK]",
        "[PAD]",
        "[RESERVED]",
        "[UNK]",
        *"abcdefghijklmnopqrstuvwxyz ",
    )
    vocab_size = 35
    mask_token_id = 4
    bos_token_id = 2
    eos_token_id = 3
    pad_token_id = 5
    unk_token_id = 7

    def __len__(self):
        return self.vocab_size

    def encode(self, text):
        ids = {symbol: index for index, symbol in enumerate(self.symbols)}
        return [ids.get(character, self.unk_token_id) for character in text.lower()]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(
            self.symbols[int(i)] if 0 <= int(i) < 35 else "[UNK]"
            for i in ids
            if not skip_special_tokens or 8 <= int(i) < 35
        )

    def word_segments(self, ids):
        # Specials separate segments; removing them must never join two words.
        segments, current = [], []
        for value in ids:
            value = int(value)
            if 8 <= value < 35:
                current.append(self.symbols[value])
            else:
                if current:
                    segments.append("".join(current))
                    current = []
        if current:
            segments.append("".join(current))
        return segments


def raw_splits(raw_dir, download=False):
    root = Path(raw_dir)
    paths = {
        split: root / f"text8.{split}.txt" for split in ("train", "validation", "test")
    }
    if all(path.exists() for path in paths.values()):
        return paths
    archive = root / "text8.zip"
    root.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        if not download:
            raise FileNotFoundError(
                f"Provide text8 split files or text8.zip in {root}, or set data.download=true"
            )
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            urllib.request.urlretrieve(
                "https://mattmahoney.net/dc/text8.zip", temporary
            )
            os.replace(temporary, archive)
        finally:
            temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(archive) as source:
        text = source.read("text8")
    if len(text) != 100_000_000:
        raise ValueError("the Text8 archive must contain 100 million characters")
    for split, start, end in (
        ("train", 0, 90_000_000),
        ("validation", 90_000_000, 95_000_000),
        ("test", 95_000_000, 100_000_000),
    ):
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as handle:
            handle.write(text[start:end])
            temporary = Path(handle.name)
        os.replace(temporary, paths[split])
    return paths


def read_characters(path):
    raw = Path(path).read_bytes()
    values = np.frombuffer(raw, dtype=np.uint8)
    if not np.all(((values >= 97) & (values <= 122)) | (values == 32)):
        raise ValueError(
            f"Text8 source must contain only lowercase a-z and spaces: {path}"
        )
    return raw


def pack_text(raw, length):
    """Legacy non-crop path: L-character chunks, append EOS, group eight, wrap.

    Group boundaries are explicit and deterministic (one preprocessing shard).
    Incomplete source chunks and each group's incomplete output are dropped.
    """
    if length <= 2:
        raise ValueError("Text8 sequence length must exceed two")
    lookup = np.full(256, 7, dtype=np.uint8)
    lookup[97:123] = np.arange(8, 34)
    lookup[32] = 34
    source = lookup[np.frombuffer(raw, dtype=np.uint8)]
    source = source[: len(source) // length * length].reshape(-1, length)
    pieces = []
    for offset in range(0, len(source), 8):
        group = source[offset : offset + 8]
        eos = np.full((len(group), 1), 3, dtype=np.uint8)
        joined = np.concatenate((group, eos), axis=1).reshape(-1)
        body = joined[: len(joined) // (length - 2) * (length - 2)].reshape(
            -1, length - 2
        )
        pieces.append(
            np.concatenate(
                (
                    np.full((len(body), 1), 2, dtype=np.uint8),
                    body,
                    np.full((len(body), 1), 3, dtype=np.uint8),
                ),
                axis=1,
            )
        )
    return np.concatenate(pieces) if pieces else np.empty((0, length), dtype=np.uint8)


class Text8Dataset(Dataset):
    def __init__(self, path, limit=None):
        self.tokens = np.load(path, mmap_mode="r")
        self.size = (
            len(self.tokens) if limit is None else min(len(self.tokens), int(limit))
        )
        if self.size <= 0:
            raise ValueError("Text8 split must contain at least one complete sequence")

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {"input_ids": torch.tensor(self.tokens[index].copy(), dtype=torch.long)}


def prepare_split(path, cache_dir, length):
    raw = read_characters(path)
    digest = hashlib.sha256(raw).hexdigest()
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"text8_{Path(path).stem}_{digest[:16]}_L{length}_pack8_v1.npy"
    if not target.exists():
        with tempfile.NamedTemporaryFile(
            dir=root, suffix=".npy", delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            np.save(temporary, pack_text(raw, length))
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return target, raw, digest


def training_vocabulary(raw, cache_dir, digest):
    path = Path(cache_dir) / f"text8_vocab_{digest[:16]}_v1.json"
    if path.exists():
        return frozenset(json.loads(path.read_text()))
    vocabulary = sorted(
        {
            match.group()
            for match in re.finditer(r"(?<= )[a-z]+(?= )", raw.decode("ascii"))
        }
    )
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        json.dump(vocabulary, handle)
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return frozenset(vocabulary)

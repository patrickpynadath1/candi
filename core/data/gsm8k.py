"""GSM8K evaluation prompts, separate from TinyGSM training data."""

import json
from pathlib import Path
import datasets
import numpy as np
from .tinygsm import _token_ids


def _records(rows):
    return [
        {"prompt": row["question"], "response_ground_truth": row["answer"]}
        for row in rows
    ]


def get_gsm8k_dataset(data, tokenizer, split="test"):
    """GSM8K prompts: the test split, or a fixed held-out slice of train.

    ``validation`` is for checkpoint and hyperparameter selection, so the test
    split is only read when it is asked for by name.
    """
    if split not in {"test", "validation"}:
        raise ValueError(f"unknown GSM8K split: {split}")
    key = "data_path" if split == "test" else "validation_data_path"
    path = getattr(data, key, None)
    if path:
        records = json.loads(Path(path).read_text())
    elif split == "test":
        records = _records(
            datasets.load_dataset("gsm8k", "main", split="test", cache_dir=data.cache_dir)
        )
    else:
        train = datasets.load_dataset(
            "gsm8k", "main", split="train", cache_dir=data.cache_dir
        )
        size = int(getattr(data, "validation_size", 500))
        seed = int(getattr(data, "validation_seed", 0))
        if not 0 < size <= len(train):
            raise ValueError("validation_size must be positive and at most the train split")
        order = np.random.default_rng(seed).permutation(len(train))[:size]
        records = _records(train.select(sorted(order.tolist())))
    bos = tokenizer.bos_token_id
    separator = _token_ids(tokenizer, data.separator)

    def tokenize(record):
        prompt = [bos] + _token_ids(tokenizer, record["prompt"].strip()) + separator
        return {**record, "input_ids": prompt}

    return (
        datasets.Dataset.from_list(records)
        .map(tokenize, desc="Tokenizing GSM8K prompts")
        .with_format("torch", columns=["input_ids"], output_all_columns=True)
    )

"""GSM8K evaluation prompts, separate from TinyGSM training data."""

import json
from pathlib import Path
import datasets
from .tinygsm import _token_ids


def get_gsm8k_dataset(data, tokenizer):
    path = getattr(data, "data_path", None)
    if path:
        records = json.loads(Path(path).read_text())
    else:
        raw = datasets.load_dataset(
            "gsm8k", "main", split="test", cache_dir=data.cache_dir
        )
        records = [
            {"prompt": row["question"], "response_ground_truth": row["answer"]}
            for row in raw
        ]
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

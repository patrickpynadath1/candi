"""TinyGSM supervised code-generation dataset preparation."""

import hashlib
import json
import itertools
from pathlib import Path

import datasets


def _token_ids(tokenizer, text):
    result = tokenizer(text, add_special_tokens=False)
    return result.input_ids if hasattr(result, "input_ids") else result["input_ids"]


def get_tinygsm_dataset(data, tokenizer, block_size):
    max_examples = getattr(data, "max_examples", None)
    streaming = getattr(data, "streaming", False)
    if streaming:
        raise ValueError("streaming TinyGSM is not supported in core yet")
    tokenizer_name = data.tokenizer_name_or_path.replace("/", "__")
    options = (
        f"prompt{int(data.train_on_prompt)}_pad{int(data.train_on_pad)}"
        f"_filter{int(data.filter_too_long)}_wrap{int(data.wrap)}"
        f"_max{max_examples if max_examples is not None else 'all'}"
        f"_val{data.val_ratio}_seed{data.val_seed}"
    )
    identity = json.dumps(
        {
            "separator": data.separator,
            "vocab": tokenizer.get_vocab(),
            "special": [
                tokenizer.bos_token_id,
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
            ],
        },
        sort_keys=True,
    )
    options += "_" + hashlib.sha256(identity.encode()).hexdigest()[:16]
    cache = Path(data.cache_dir) / f"tiny_gsm_bs{block_size}_{options}_{tokenizer_name}"
    if getattr(data, "data_path", None):
        source_hash = hashlib.sha256(Path(data.data_path).read_bytes()).hexdigest()[:16]
        cache = cache.with_name(cache.name + "_" + source_hash)
    if cache.exists():
        return datasets.load_from_disk(str(cache)).with_format("torch")

    if getattr(data, "data_path", None):
        raw = datasets.Dataset.from_list(json.loads(Path(data.data_path).read_text()))
        if max_examples is not None:
            raw = raw.select(range(min(max_examples, len(raw))))
    elif max_examples is None and not streaming:
        raw = datasets.load_dataset(
            "TinyGSM/TinyGSM", split="train", cache_dir=data.cache_dir
        )
    elif max_examples is not None:
        if max_examples < 2:
            raise ValueError("TinyGSM max_examples must be at least 2")
        streamed = datasets.load_dataset(
            "TinyGSM/TinyGSM",
            split="train",
            cache_dir=data.cache_dir,
            streaming=True,
        )
        raw = datasets.Dataset.from_list(list(itertools.islice(streamed, max_examples)))
    bos, eos, pad = (
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    )
    separator = _token_ids(tokenizer, data.separator)

    def tokenize(example):
        question = _token_ids(tokenizer, example["question"].strip())
        answer = _token_ids(tokenizer, example["code"].strip())
        ids = [bos] + question + separator + answer + [eos]
        return {"input_ids": ids, "prompt_len": 1 + len(question) + len(separator)}

    def pad_and_mask(example):
        original = example["input_ids"]
        length = min(len(original), block_size)
        ids = original[:block_size]
        if len(original) > block_size:
            ids[-1] = eos
        ids += [pad] * (block_size - len(ids))
        mask_start = (
            0 if data.train_on_prompt else min(example["prompt_len"], block_size)
        )
        mask_end = block_size if data.train_on_pad else length
        mask = (
            [0] * mask_start
            + [1] * max(0, mask_end - mask_start)
            + [0] * (block_size - mask_end)
        )
        return {"input_ids": ids, "attention_mask": mask}

    tokenized = raw.map(
        tokenize, remove_columns=raw.column_names, desc="Tokenizing TinyGSM"
    )
    if data.filter_too_long:
        if data.wrap:
            raise ValueError("filter_too_long and wrap cannot both be enabled")
        tokenized = tokenized.filter(lambda row: len(row["input_ids"]) <= block_size)

    if data.wrap:
        tokenized = tokenized.remove_columns("prompt_len")

        def wrap_batch(batch):
            merged = list(itertools.chain.from_iterable(batch["input_ids"]))
            usable = len(merged) // block_size * block_size
            chunks = [
                merged[index : index + block_size]
                for index in range(0, usable, block_size)
            ]
            return {
                "input_ids": chunks,
                "attention_mask": [[1] * block_size for _ in chunks],
            }

        tokenized = tokenized.map(
            wrap_batch,
            batched=True,
            batch_size=1000,
            remove_columns=tokenized.column_names,
            desc="Wrapping TinyGSM",
        )
    else:
        tokenized = tokenized.map(
            pad_and_mask, remove_columns=["prompt_len"], desc="Padding TinyGSM"
        )

    split = tokenized.train_test_split(test_size=data.val_ratio, seed=data.val_seed)
    dataset = datasets.DatasetDict(
        {"train": split["train"], "validation": split["test"]}
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(cache))
    return dataset.with_format("torch")

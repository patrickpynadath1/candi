"""TinyGSM supervised code-generation dataset preparation."""

import itertools
from pathlib import Path

import datasets


def _token_ids(tokenizer, text):
    result = tokenizer(text, add_special_tokens=False)
    return result.input_ids if hasattr(result, "input_ids") else result["input_ids"]


def get_tinygsm_dataset(config, tokenizer):
    data, block_size = config.data, config.model.length
    tokenizer_name = data.tokenizer_name_or_path.replace("/", "__")
    options = (
        f"prompt{int(data.train_on_prompt)}_pad{int(data.train_on_pad)}"
        f"_filter{int(data.filter_too_long)}_wrap{int(data.wrap)}"
    )
    cache = Path(data.cache_dir) / f"tiny_gsm_bs{block_size}_{options}_{tokenizer_name}"
    if cache.exists():
        return datasets.load_from_disk(str(cache)).with_format("torch")

    raw = datasets.load_dataset(
        "TinyGSM/TinyGSM", split="train", cache_dir=data.cache_dir
    )
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

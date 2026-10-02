"""OpenWebText document splits and legacy eight-document sequence packing."""


def pack_documents(examples, length, bos, eos):
    joined = [token for document in examples["input_ids"] for token in document]
    body = length - 2
    usable = len(joined) // body * body
    return {
        "input_ids": [
            [bos] + joined[i : i + body] + [eos] for i in range(0, usable, body)
        ]
    }


def prepare_openwebtext(data, tokenizer, length, split):
    from datasets import load_dataset, load_from_disk

    if split not in {"train", "validation"}:
        raise ValueError(f"unknown OpenWebText split: {split}")
    prepared = data.get("prepared_" + split)
    if prepared:
        dataset = load_from_disk(prepared)
    else:
        files = data.get("files_" + split)
        if files:
            dataset = load_dataset(
                "json", data_files=list(files), split="train", cache_dir=data.cache_dir
            )
        else:
            dataset = load_dataset(
                data.dataset_name,
                split=data[split + "_split"],
                cache_dir=data.cache_dir,
                revision=data.get("revision"),
            )
        limit = data.get("limit_documents_" + split)
        if limit is not None:
            dataset = dataset.select(range(min(len(dataset), int(limit))))

        def tokenize(examples):
            encoded = tokenizer(
                examples["text"],
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            return {
                "input_ids": [
                    ids + [tokenizer.eos_token_id] for ids in encoded["input_ids"]
                ]
            }

        dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=data.num_proc if data.num_proc > 1 else None,
        )
        dataset = dataset.map(
            pack_documents,
            batched=True,
            batch_size=8,
            fn_kwargs={
                "length": length,
                "bos": tokenizer.bos_token_id,
                "eos": tokenizer.eos_token_id,
            },
            remove_columns=dataset.column_names,
            num_proc=data.num_proc if data.num_proc > 1 else None,
        )
    limit = data.get("limit_" + split)
    if limit is not None:
        dataset = dataset.select(range(min(len(dataset), int(limit))))
    if not len(dataset):
        raise ValueError("OpenWebText split has no complete sequences")
    if len(dataset[0]["input_ids"]) != length:
        raise ValueError(
            "prepared OpenWebText sequence length does not match task.length"
        )
    return dataset.with_format("torch", columns=["input_ids"])

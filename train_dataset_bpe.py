#!/usr/bin/env python
import argparse
import json
import os
import re
import shutil
import sys

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast


DEFAULT_SPECIAL_TOKENS = [
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[PAD]",
    "[MASK]",
    "[BOS]",
    "[EOS]",
]

DEFAULT_VOCAB_SIZES = [128, 1024, 8192, 16384]
DEFAULT_BATCH_SIZE = 128
WIKIPEDIA_EN_CLEAN_DATASET_NAME = "wikipedia_en_clean"


def _get_dataloader_exports():
    from dataloader import (
        get_text8_splits,
        get_tiny_shakespeare_splits,
        get_wikipedia_en_clean_dataset,
        prepare_wikipedia_en_clean,
    )

    return {
        "get_text8_splits": get_text8_splits,
        "get_tiny_shakespeare_splits": get_tiny_shakespeare_splits,
        "get_wikipedia_en_clean_dataset": get_wikipedia_en_clean_dataset,
        "prepare_wikipedia_en_clean": prepare_wikipedia_en_clean,
    }


def _text8_info(cache_dir):
    splits = _get_dataloader_exports()["get_text8_splits"](cache_dir)
    dataset_cache_dir = os.path.join(cache_dir, "text8")
    return {
        "splits": splits,
        "dataset_cache_dir": dataset_cache_dir,
        "train_path": os.path.join(dataset_cache_dir, "raw_data", "text8.train.txt"),
    }


def _tiny_shakespeare_info(cache_dir):
    splits = _get_dataloader_exports()["get_tiny_shakespeare_splits"](cache_dir)
    dataset_cache_dir = os.path.join(cache_dir, "tiny_shakespeare")
    return {
        "splits": splits,
        "dataset_cache_dir": dataset_cache_dir,
        "train_path": os.path.join(
            dataset_cache_dir, "raw_data", "tiny_shakespeare.train.txt"
        ),
    }


def _wikipedia_en_clean_info(cache_dir):
    dataloader_exports = _get_dataloader_exports()
    metadata = dataloader_exports["prepare_wikipedia_en_clean"](cache_dir)
    dataset = dataloader_exports["get_wikipedia_en_clean_dataset"](cache_dir)
    train_dataset = dataset["train"]
    split_metadata = metadata["splits"]
    return {
        "metadata": metadata,
        "train_dataset": train_dataset,
        "dataset_cache_dir": os.path.join(cache_dir, WIKIPEDIA_EN_CLEAN_DATASET_NAME),
        "split_stats": {
            "train_articles": split_metadata["train"]["article_count"],
            "validation_articles": split_metadata["validation"]["article_count"],
            "test_articles": split_metadata["test"]["article_count"],
            "train_bytes": split_metadata["train"]["raw_text_bytes"],
            "validation_bytes": split_metadata["validation"]["raw_text_bytes"],
            "test_bytes": split_metadata["test"]["raw_text_bytes"],
        },
    }


DATASET_PREPARERS = {
    "text8": {
        "default_cache_dir": "/home/patrick/.cache/discrete_diffusion/text8",
        "loader": _text8_info,
    },
    "tiny_shakespeare": {
        "default_cache_dir": "/home/patrick/.cache/discrete_diffusion/tiny_shakespeare",
        "loader": _tiny_shakespeare_info,
    },
    WIKIPEDIA_EN_CLEAN_DATASET_NAME: {
        "default_cache_dir": "/home/patrick/.cache/discrete_diffusion/wikipedia_en_clean",
        "loader": _wikipedia_en_clean_info,
    },
}


def _parse_vocab_sizes(values):
    if not values:
        return list(DEFAULT_VOCAB_SIZES)
    vocab_sizes = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                vocab_sizes.append(int(item))
    if not vocab_sizes:
        raise ValueError("At least one vocab size must be provided.")
    return vocab_sizes


def _get_minimum_vocab_size(train_path, special_tokens):
    with open(train_path, "rb") as f:
        unique_bytes = set(f.read())
    return len(unique_bytes) + len(special_tokens)


def _get_minimum_vocab_size_from_dataset(train_dataset, special_tokens):
    unique_bytes = set()
    for text in tqdm(
        train_dataset["text"],
        desc="Scanning dataset bytes",
        unit="article",
    ):
        unique_bytes.update(text.encode("utf-8"))
    return len(unique_bytes) + len(special_tokens)


def _build_tokenizer(
    vocab_size,
    min_frequency,
    special_tokens,
    train_path=None,
    text_iterator=None,
    length=None,
):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=special_tokens,
    )
    if train_path is not None:
        tokenizer.train([train_path], trainer=trainer)
    else:
        tokenizer.train_from_iterator(text_iterator, trainer=trainer, length=length)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [SEP] $B [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        unk_token="[UNK]",
    )


def _iter_batches(seq, batch_size):
    batch = []
    for item in seq:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _iter_text_batches(train_dataset, batch_size):
    total_batches = (len(train_dataset) + batch_size - 1) // batch_size
    for batch in tqdm(
        _iter_batches(train_dataset, batch_size),
        total=total_batches,
        desc="Streaming text batches to tokenizer trainer",
        unit="batch",
    ):
        yield [row["text"] for row in batch if row["text"]]


def _slugify_tokenizer_name(tokenizer_name_or_path):
    slug = tokenizer_name_or_path.rstrip(os.sep)
    slug = slug.replace("\\", "/")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", slug)
    slug = slug.strip("._-")
    return slug or "tokenizer"


def _prepare_output_dir(save_dir, overwrite):
    if os.path.exists(save_dir):
        if not overwrite:
            raise FileExistsError(
                f"{save_dir} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(save_dir)


def _normalize_imported_tokenizer(tokenizer):
    special_tokens_map = tokenizer.special_tokens_map
    bos_token = special_tokens_map.get("bos_token")
    cls_token = special_tokens_map.get("cls_token")
    eos_token = special_tokens_map.get("eos_token")
    sep_token = special_tokens_map.get("sep_token")
    pad_token = special_tokens_map.get("pad_token")

    if bos_token is None:
        if cls_token is None:
            raise AttributeError(
                "Imported tokenizer must define a bos_token or cls_token."
            )
        tokenizer.bos_token = cls_token
    if eos_token is None:
        if sep_token is None:
            raise AttributeError(
                "Imported tokenizer must define an eos_token or sep_token."
            )
        tokenizer.eos_token = sep_token
    if pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def _build_metadata_base(args, info):
    metadata = {
        "dataset": args.dataset,
        "dataset_cache_dir": info["dataset_cache_dir"],
        "source_split": "train",
    }
    train_path = info.get("train_path")
    if train_path is not None:
        metadata["train_path"] = train_path
    if "split_stats" in info:
        metadata.update(info["split_stats"])
    elif "splits" in info:
        metadata.update(
            {
                "train_characters": len(info["splits"]["train"]),
                "validation_characters": len(info["splits"]["validation"]),
                "test_characters": len(info["splits"]["test"]),
            }
        )
    return metadata


def _get_lightweight_dataset_info(dataset, cache_dir):
    if dataset == "text8":
        dataset_cache_dir = os.path.join(cache_dir, "text8")
        return {
            "dataset_cache_dir": dataset_cache_dir,
            "train_path": os.path.join(dataset_cache_dir, "raw_data", "text8.train.txt"),
        }
    if dataset == "tiny_shakespeare":
        dataset_cache_dir = os.path.join(cache_dir, "tiny_shakespeare")
        return {
            "dataset_cache_dir": dataset_cache_dir,
            "train_path": os.path.join(
                dataset_cache_dir, "raw_data", "tiny_shakespeare.train.txt"
            ),
        }
    return {
        "dataset_cache_dir": os.path.join(cache_dir, WIKIPEDIA_EN_CLEAN_DATASET_NAME),
    }


def _write_training_metadata(save_dir, metadata):
    with open(os.path.join(save_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def _train_bpe_tokenizers(args, info, output_root):
    train_path = info.get("train_path")
    train_dataset = info.get("train_dataset")
    vocab_sizes = sorted(_parse_vocab_sizes(args.vocab_sizes))
    if train_dataset is not None:
        minimum_vocab_size = _get_minimum_vocab_size_from_dataset(
            train_dataset, args.special_tokens
        )
    else:
        minimum_vocab_size = _get_minimum_vocab_size(train_path, args.special_tokens)

    metadata_base = _build_metadata_base(args, info)
    for vocab_size in vocab_sizes:
        if vocab_size < minimum_vocab_size:
            raise ValueError(
                "Requested vocab size {} is smaller than the minimum achievable "
                "size {} for the current ByteLevel BPE setup ({} unique bytes + "
                "{} special tokens).".format(
                    vocab_size,
                    minimum_vocab_size,
                    minimum_vocab_size - len(args.special_tokens),
                    len(args.special_tokens),
                )
            )

        save_dir = os.path.join(output_root, f"bpe_vocab{vocab_size}")
        _prepare_output_dir(save_dir, args.overwrite)

        if train_dataset is not None:
            hf_tokenizer = _build_tokenizer(
                vocab_size=vocab_size,
                min_frequency=args.min_frequency,
                special_tokens=args.special_tokens,
                text_iterator=_iter_text_batches(train_dataset, args.batch_size),
                length=len(train_dataset),
            )
        else:
            hf_tokenizer = _build_tokenizer(
                vocab_size=vocab_size,
                min_frequency=args.min_frequency,
                special_tokens=args.special_tokens,
                train_path=train_path,
            )
        hf_tokenizer.save_pretrained(save_dir)

        metadata = {
            **metadata_base,
            "tokenizer_source": "train",
            "tokenizer_type": "bytelevel_bpe",
            "minimum_vocab_size": minimum_vocab_size,
            "vocab_size": vocab_size,
            "min_frequency": args.min_frequency,
            "special_tokens": args.special_tokens,
        }
        _write_training_metadata(save_dir, metadata)
        print(f"Saved tokenizer to {save_dir}")


def _import_tokenizer(args, info, output_root):
    tokenizer_name_or_path = args.import_tokenizer_name_or_path
    if not tokenizer_name_or_path:
        raise ValueError(
            "--import-tokenizer-name-or-path is required when --tokenizer-source=import."
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    tokenizer = _normalize_imported_tokenizer(tokenizer)

    save_dir = os.path.join(
        output_root, f"imported_{_slugify_tokenizer_name(tokenizer_name_or_path)}"
    )
    _prepare_output_dir(save_dir, args.overwrite)
    tokenizer.save_pretrained(save_dir)

    metadata = {
        **_build_metadata_base(args, info),
        "tokenizer_source": "import",
        "tokenizer_type": getattr(tokenizer, "tokenizer_type", None),
        "imported_from": tokenizer_name_or_path,
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "special_tokens": tokenizer.special_tokens_map,
    }
    _write_training_metadata(save_dir, metadata)
    print(f"Saved tokenizer to {save_dir}")


def _validate_args(args):
    argv = sys.argv[1:]
    if args.tokenizer_source == "import":
        if args.vocab_sizes is not None:
            raise ValueError("--vocab-sizes can only be used when --tokenizer-source=train.")
        if "--min-frequency" in argv:
            raise ValueError(
                "--min-frequency can only be used when --tokenizer-source=train."
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train Hugging Face ByteLevel BPE tokenizers or import external "
            "Hugging Face tokenizers for cached text datasets."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_PREPARERS.keys()),
        default="text8",
        help="Dataset to prepare and associate with the saved tokenizer.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Dataset cache root used by the dataset dataloader helper.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for saved tokenizer artifacts. Defaults under the cache dir.",
    )
    parser.add_argument(
        "--tokenizer-source",
        choices=["train", "import"],
        default="train",
        help="Whether to train a ByteLevel BPE tokenizer or import an external one.",
    )
    parser.add_argument(
        "--import-tokenizer-name-or-path",
        default=None,
        help="Hugging Face tokenizer name or local path to import when tokenizer-source=import.",
    )
    parser.add_argument(
        "--vocab-sizes",
        nargs="+",
        default=None,
        help="One or more vocab sizes for BPE training. Supports repeated values or comma-separated lists.",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum token frequency for BPE merges.",
    )
    parser.add_argument(
        "--special-tokens",
        nargs="+",
        default=DEFAULT_SPECIAL_TOKENS,
        help="Special tokens to reserve in the trained BPE tokenizer.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tokenizer directories.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for streaming dataset-backed BPE training.",
    )
    args = parser.parse_args()
    _validate_args(args)

    dataset_info = DATASET_PREPARERS[args.dataset]
    cache_dir = args.cache_dir or dataset_info["default_cache_dir"]
    if args.tokenizer_source == "train":
        info = dataset_info["loader"](cache_dir)
        train_path = info.get("train_path")
        if train_path is not None and not os.path.exists(train_path):
            raise FileNotFoundError(
                f"Expected {args.dataset} train split at {train_path}"
            )
    else:
        info = _get_lightweight_dataset_info(args.dataset, cache_dir)

    output_root = args.output_dir
    if output_root is None:
        output_root = os.path.join(cache_dir, "tokenizers", args.dataset)
    os.makedirs(output_root, exist_ok=True)

    if args.tokenizer_source == "train":
        _train_bpe_tokenizers(args, info, output_root)
    else:
        _import_tokenizer(args, info, output_root)


if __name__ == "__main__":
    main()

import functools
import hashlib
import pdb
import itertools
import json
import math
import os
import re
import shutil
import typing
import urllib
import zipfile
from typing import Optional

import datasets
import fsspec
import requests
import tokenizers
import torch
import transformers
from tqdm.auto import tqdm

import utils
from tasks import get_task_dataset, is_task_dataset

LOGGER = utils.get_logger(__name__)

WIKIPEDIA_EN_CLEAN_DATASET_NAME = "wikipedia_en_clean"
WIKIPEDIA_EN_CLEAN_HF_DATASET = "wikimedia/wikipedia"
WIKIPEDIA_EN_CLEAN_HF_CONFIG = "20231101.en"
WIKIPEDIA_EN_CLEAN_SHUFFLE_SEED = 1337
WIKIPEDIA_EN_CLEAN_TRAIN_SPLIT = 0.98
WIKIPEDIA_EN_CLEAN_VALID_SPLIT = 0.01
WIKIPEDIA_EN_CLEAN_TRAIN_BYTE_CAP = 20 * 1024**3
WIKIPEDIA_EN_CLEAN_NORMALIZATION = "text8_like_v1"


def _get_tokenizer_cache_fingerprint(tokenizer):
    digest = hashlib.sha256()
    digest.update(tokenizer.__class__.__name__.encode("utf-8"))
    digest.update(str(getattr(tokenizer, "name_or_path", "")).encode("utf-8"))
    digest.update(str(tokenizer.vocab_size).encode("utf-8"))
    digest.update(str(len(tokenizer)).encode("utf-8"))
    digest.update(",".join(tokenizer.all_special_tokens).encode("utf-8"))
    digest.update(",".join(map(str, tokenizer.all_special_ids)).encode("utf-8"))

    # Include the realized vocabulary so tokenizer swaps or local tokenizer
    # edits do not silently reuse an incompatible tokenized dataset cache.
    vocab = tokenizer.get_vocab()
    if hasattr(tokenizer, "get_added_vocab"):
        vocab = {**vocab, **tokenizer.get_added_vocab()}
    for token, index in sorted(vocab.items(), key=lambda item: (item[1], item[0])):
        digest.update(token.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
        digest.update(str(index).encode("utf-8"))
        digest.update(b"\n")

    return digest.hexdigest()[:12]


def wt_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def ptb_detokenizer(x):
    x = x.replace(" 's", "'s")
    x = x.replace("s ' ", "s' ")
    x = x.replace(" n't", "n't")
    x = x.replace(" \n ", "\n")
    x = x.replace("\\/", "/")
    for _ in range(10):
        x = x.replace(" N ", " 1 ")
    x = x.replace("$ 1", "$1")
    x = x.replace("# 1", "#1")
    x = x.replace("<unk>", "?")
    return x


def lm1b_detokenizer(x):
    x = x.replace("http : / / ", "http://")
    x = x.replace("https : / / ", "https://")
    x = re.sub(r" \'(\w+)", r"'\1", x)
    x = re.sub(r" (\w+) \. ", r" \1. ", x)
    x = re.sub(r" (\w+) \.$", r" \1.", x)
    x = x.replace(" ? ", "? ")
    x = re.sub(r" \?$", "?", x)
    x = x.replace(" ! ", "! ")
    x = re.sub(r" \!$", "!", x)
    x = x.replace(" , ", ", ")
    x = x.replace(" : ", ": ")
    x = x.replace(" ; ", "; ")
    x = x.replace(" / ", "/")
    x = re.sub(r"\" ([^\"]+) \"", r'"\1"', x)
    x = re.sub(r"\' ([^\']+) \'", r"'\1'", x)
    x = re.sub(r"\( ([^\(\)]+) \)", r"(\1)", x)
    x = re.sub(r"\[ ([^\[\]]+) \]", r"[\1]", x)
    x = x.replace("$ ", "$")
    x = x.replace("£ ", "£")
    return x


def lambada_detokenizer(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return "\n" + text.strip()


def scientific_papers_detokenizer(x):
    x = wt_detokenizer(x)
    x = lm1b_detokenizer(x)
    return x


class Text8Tokenizer(transformers.PreTrainedTokenizer):
    def __init__(
        self,
        bos_token="[BOS]",
        eos_token="[EOS]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        unk_token="[UNK]",
        **kwargs,
    ):
        self.characters = list("abcdefghijklmnopqrstuvwxyz ")
        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[EOS]": 3,
            "[MASK]": 4,
            "[PAD]": 5,
            "[RESERVED]": 6,
            "[UNK]": 7,
            **{ch: i + 8 for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
        return list(text.lower())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def get_vocab(self) -> typing.Dict[str, int]:
        return self._vocab_str_to_int


def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
        response = requests.get(url, stream=True)
        data_list = []

        # Process each line in the response content
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                data_list.append(data)

        return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = datasets.Dataset.from_list(lambada_data)
    return dataset


def get_text8_splits(cache_dir):
    url = "http://mattmahoney.net/dc/text8.zip"
    cache_dir = f"{cache_dir}/text8"
    split_names = ["train", "validation", "test"]
    raw_cache_dir = os.path.join(cache_dir, "raw_data")
    if not all(
        [
            utils.fsspec_exists(os.path.join(raw_cache_dir, f"text8.{split}.txt"))
            for split in split_names
        ]
    ):
        if not utils.fsspec_exists(os.path.join(raw_cache_dir, "text8.zip")):
            utils.fsspec_mkdirs(raw_cache_dir, exist_ok=True)
            LOGGER.info("Downloading text8 from URL {}.".format(url))
            with (
                urllib.request.urlopen(url) as in_stream,
                open(os.path.join(raw_cache_dir, "text8.zip"), "wb") as out_file,
            ):
                shutil.copyfileobj(in_stream, out_file)

        with fsspec.open(os.path.join(raw_cache_dir, "text8.zip"), "rb") as f:
            rawdata = zipfile.ZipFile(f).read("text8").decode("utf-8")

        # Splits taken from D3PM codebase
        splits = {
            "train": rawdata[:90000000],
            "validation": rawdata[90000000:95000000],
            "test": rawdata[95000000:],
        }

        for split, data in splits.items():
            _path = os.path.join(raw_cache_dir, f"text8.{split}.txt")
            with fsspec.open(_path, "w") as f:
                f.write(data)
    else:
        splits = {}
        for split in split_names:
            _path = os.path.join(raw_cache_dir, f"text8.{split}.txt")
            with fsspec.open(_path, "r") as f:
                splits[split] = f.read()

    return splits


def get_text8_dataset(cache_dir, max_seq_length=256, drop_last=True, crop_train=False):
    """Adapted from:
    https://github.com/google-research/google-research/blob/master/d3pm/text/datasets.py#L344

    Args:
      cache_dir: str, path to cache directory.
      max_seq_length: int, maximum length of sequences.
          (default: 256, as in D3PM codebase.)
      drop_last: bool, whether to drop the last incomplete
          batch. (default: True, as in D3PM codebase.)
      crop_train: bool, whether to subsample contiguous
          subsequences from training example. serves to
          make sure transformer models with absolute position
          embeddings do not have incorrect position-wise
          marginals. (default: False, but necessary to match D3PM AR)

    Returns:
      dataset: dataset.DatasetDict, with keys 'train',
          'valid', 'test'.
    """
    if not crop_train:
        cache_dir = f"{cache_dir}/text8"
        splits = None
    else:
        cache_dir = f"{cache_dir}/text8-crop-train"
        splits = get_text8_splits(os.path.dirname(cache_dir))
    split_names = ["train", "validation", "test"]
    if not all(
        [utils.fsspec_exists(os.path.join(cache_dir, split)) for split in split_names]
    ):
        if splits is None:
            splits = get_text8_splits(os.path.dirname(cache_dir))

        # Chunk and save as datasets.DatasetDict
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        dataset_dict = {}
        for k, v in splits.items():
            if k == "train" and crop_train == True:
                chunk_size = 2 * max_seq_length
            else:
                chunk_size = max_seq_length
            text = list(chunks(v, chunk_size))
            if drop_last and len(text[-1]) < chunk_size:
                text = text[:-1]
            dataset_dict[k] = datasets.Dataset.from_dict({"text": text})
        dataset = datasets.DatasetDict(dataset_dict)
        dataset.save_to_disk(cache_dir)
    else:
        dataset = datasets.load_from_disk(cache_dir)

    return dataset


def get_tiny_shakespeare_splits(cache_dir):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    dataset_cache_dir = os.path.join(cache_dir, "tiny_shakespeare")
    raw_cache_dir = os.path.join(dataset_cache_dir, "raw_data")
    split_names = ["train", "validation", "test"]

    split_paths = {
        split: os.path.join(raw_cache_dir, f"tiny_shakespeare.{split}.txt")
        for split in split_names
    }
    if not all(utils.fsspec_exists(path) for path in split_paths.values()):
        utils.fsspec_mkdirs(raw_cache_dir, exist_ok=True)
        raw_path = os.path.join(raw_cache_dir, "input.txt")
        if not utils.fsspec_exists(raw_path):
            LOGGER.info("Downloading Tiny Shakespeare from URL {}.".format(url))
            with (
                urllib.request.urlopen(url) as in_stream,
                fsspec.open(raw_path, "wb") as out_file,
            ):
                shutil.copyfileobj(in_stream, out_file)

        with fsspec.open(raw_path, "r") as f:
            rawdata = f.read()

        total_length = len(rawdata)
        train_end = int(total_length * 0.9)
        validation_end = int(total_length * 0.95)
        splits = {
            "train": rawdata[:train_end],
            "validation": rawdata[train_end:validation_end],
            "test": rawdata[validation_end:],
        }

        for split, data in splits.items():
            with fsspec.open(split_paths[split], "w") as f:
                f.write(data)
    else:
        splits = {}
        for split, path in split_paths.items():
            with fsspec.open(path, "r") as f:
                splits[split] = f.read()

    return splits


def get_tiny_shakespeare_dataset(cache_dir, max_seq_length=256, drop_last=True, crop_train=False):
    del crop_train  # Tiny Shakespeare uses a single deterministic chunking path.
    dataset_cache_dir = os.path.join(cache_dir, "tiny_shakespeare")
    split_names = ["train", "validation", "test"]
    if not all(
        utils.fsspec_exists(os.path.join(dataset_cache_dir, split))
        for split in split_names
    ):
        splits = get_tiny_shakespeare_splits(cache_dir)

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        dataset_dict = {}
        for split, text in splits.items():
            chunked_text = list(chunks(text, max_seq_length))
            if drop_last and chunked_text and len(chunked_text[-1]) < max_seq_length:
                chunked_text = chunked_text[:-1]
            dataset_dict[split] = datasets.Dataset.from_dict({"text": chunked_text})
        dataset = datasets.DatasetDict(dataset_dict)
        dataset.save_to_disk(dataset_cache_dir)
    else:
        dataset = datasets.load_from_disk(dataset_cache_dir)

    return dataset


def _get_wikipedia_en_clean_paths(cache_dir):
    dataset_cache_dir = os.path.join(cache_dir, WIKIPEDIA_EN_CLEAN_DATASET_NAME)
    split_dataset_dir = os.path.join(dataset_cache_dir, "article_splits")
    return {
        "dataset_cache_dir": dataset_cache_dir,
        "split_dataset_dir": split_dataset_dir,
        "metadata_path": os.path.join(dataset_cache_dir, "prep_metadata.json"),
    }


def normalize_text8_like(text):
    text = text.lower()
    text = re.sub(r"[^a-z]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_wikipedia_en_clean(cache_dir):
    paths = _get_wikipedia_en_clean_paths(cache_dir)
    if all(
        utils.fsspec_exists(path)
        for path in [paths["metadata_path"], paths["split_dataset_dir"]]
    ):
        with fsspec.open(paths["metadata_path"], "r") as f:
            metadata = json.load(f)
        if metadata.get("normalization") == WIKIPEDIA_EN_CLEAN_NORMALIZATION:
            return metadata

    if utils.fsspec_exists(paths["split_dataset_dir"]):
        shutil.rmtree(paths["split_dataset_dir"])

    dataset = datasets.load_dataset(
        WIKIPEDIA_EN_CLEAN_HF_DATASET,
        WIKIPEDIA_EN_CLEAN_HF_CONFIG,
        split="train",
        cache_dir=cache_dir,
    )
    dataset = dataset.shuffle(seed=WIKIPEDIA_EN_CLEAN_SHUFFLE_SEED)
    total_articles = len(dataset)
    train_count = int(total_articles * WIKIPEDIA_EN_CLEAN_TRAIN_SPLIT)
    validation_count = int(total_articles * WIKIPEDIA_EN_CLEAN_VALID_SPLIT)
    split_ranges = {
        "train": (0, train_count),
        "validation": (train_count, train_count + validation_count),
        "test": (train_count + validation_count, total_articles),
    }
    split_metadata = {}
    split_datasets = {}

    def _normalize_examples(examples):
        texts = [normalize_text8_like(text) for text in examples["text"]]
        return {
            "text": texts,
            "byte_length": [len(text.encode("utf-8")) for text in texts],
        }

    for split, (start, end) in split_ranges.items():
        split_source = dataset.select(range(start, end))
        kept_indices = []
        article_count = 0
        raw_text_bytes = 0
        for local_index, example in enumerate(
            tqdm(
                split_source,
                desc=f"Scanning {split} articles",
                unit="article",
            )
        ):
            text = normalize_text8_like(example["text"])
            if not text:
                continue
            byte_length = len(text.encode("utf-8"))
            if (
                split == "train"
                and article_count > 0
                and raw_text_bytes + byte_length > WIKIPEDIA_EN_CLEAN_TRAIN_BYTE_CAP
            ):
                break
            kept_indices.append(local_index)
            article_count += 1
            raw_text_bytes += byte_length

        subset = split_source.select(kept_indices)
        subset = subset.map(
            _normalize_examples,
            batched=True,
            remove_columns=subset.column_names,
            desc=f"Adding byte lengths for {split}",
        )
        split_datasets[split] = subset

        split_metadata[split] = {
            "start_index": start,
            "end_index": end,
            "article_count": article_count,
            "raw_text_bytes": raw_text_bytes,
        }

    dataset_dict = datasets.DatasetDict(split_datasets)
    dataset_dict.save_to_disk(paths["split_dataset_dir"])

    metadata = {
        "dataset_name": WIKIPEDIA_EN_CLEAN_DATASET_NAME,
        "hf_dataset": WIKIPEDIA_EN_CLEAN_HF_DATASET,
        "hf_config": WIKIPEDIA_EN_CLEAN_HF_CONFIG,
        "shuffle_seed": WIKIPEDIA_EN_CLEAN_SHUFFLE_SEED,
        "train_split_ratio": WIKIPEDIA_EN_CLEAN_TRAIN_SPLIT,
        "validation_split_ratio": WIKIPEDIA_EN_CLEAN_VALID_SPLIT,
        "test_split_ratio": 1.0
        - WIKIPEDIA_EN_CLEAN_TRAIN_SPLIT
        - WIKIPEDIA_EN_CLEAN_VALID_SPLIT,
        "train_byte_cap": WIKIPEDIA_EN_CLEAN_TRAIN_BYTE_CAP,
        "normalization": WIKIPEDIA_EN_CLEAN_NORMALIZATION,
        "source_total_articles": total_articles,
        "splits": split_metadata,
    }
    with fsspec.open(paths["metadata_path"], "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def get_wikipedia_en_clean_dataset(cache_dir):
    paths = _get_wikipedia_en_clean_paths(cache_dir)
    prepare_wikipedia_en_clean(cache_dir)
    return datasets.load_from_disk(paths["split_dataset_dir"])


def get_article_limit_from_manifest(manifest_path):
    with fsspec.open(manifest_path, "r") as f:
        manifest = json.load(f)
    return manifest["article_count_cutoff"]


def get_algorithmic_creativity_dataset(cache_dir, tokenizer, max_seq_length=256, drop_last=True, crop_train=False):

    train_path = f"{cache_dir}/train.json"
    valid_path = f"{cache_dir}/valid.json"
    test_path = f"{cache_dir}/test.json"
    
    train_data = json.load(open(train_path, 'r'))
    valid_data = json.load(open(valid_path, 'r'))
    test_data = json.load(open(test_path, 'r'))
    vocab = json.load(open(f"{cache_dir}/vocab.json", 'r'))
    tokenizer.add_tokens(vocab)

    # add vocab to the tokenizer 

    dataset = datasets.DatasetDict({
        "train": datasets.Dataset.from_list(train_data),
        "validation": datasets.Dataset.from_list(valid_data),
        "test": datasets.Dataset.from_list(test_data),
    })
    return dataset, tokenizer, vocab



def _group_texts(examples, block_size, bos, eos, insert_special_tokens=True):
    # Concatenate all texts.
    concatenated_examples = list(itertools.chain(*examples["input_ids"]))
    total_length = len(concatenated_examples)
    # TODO(yair): look into not dropping the remainder but rather padding it.
    # We drop the small remainder, and if the total_length < block_size - 2
    # we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of
    # this drop, you can customize this part to your needs.
    if insert_special_tokens:
        if block_size <= 2:
            raise ValueError("block_size must be greater than 2 when adding BOS/EOS")
        new_block_size = block_size - 2  # [BOS] and [EOS] to be added
    else:
        new_block_size = block_size
    total_length = (total_length // new_block_size) * new_block_size
    # Split by chunks of max_len.
    result = {}
    _values = []
    _attn_masks = []
    for i in range(0, total_length, new_block_size):
        if insert_special_tokens:
            _values.append(
                [bos] + concatenated_examples[i : i + new_block_size] + [eos]
            )
        else:
            _values.append(concatenated_examples[i : i + new_block_size])
        _attn_masks.append(torch.ones(block_size))
    result["input_ids"] = _values
    result["attention_mask"] = _attn_masks
    return result


def get_dataset(
    dataset_name,
    tokenizer,
    wrap,
    mode,
    cache_dir,
    block_size=1024,
    num_proc=len(os.sched_getaffinity(0)),
    streaming=False,
    revision: Optional[str] = None,
    insert_eos=True,
    insert_special_tokens=True,
    article_limit: Optional[int] = None,
):
    tokenizer_tag = _get_tokenizer_cache_fingerprint(tokenizer)
    eos_tag = ""
    if not insert_eos:
        eos_tag = "_eosFalse"
    if not insert_special_tokens:
        eos_tag = "_specialFalse"
    subset_tag = ""
    if article_limit is not None:
        subset_tag = f"_n{article_limit}"

    if dataset_name == "openwebtext-train":
        pass
    if wrap:
        filename = (
            f"{dataset_name}_{mode}_bs{block_size}_wrapped"
            f"_tok{tokenizer_tag}{subset_tag}{eos_tag}.dat"
        )
    else:
        filename = (
            f"{dataset_name}_{mode}_bs{block_size}_unwrapped"
            f"_tok{tokenizer_tag}{subset_tag}{eos_tag}.dat"
        )
    _path = os.path.join(cache_dir, filename)

    if utils.fsspec_exists(_path):
        LOGGER.info(f"Loading data from: {_path}")
        return datasets.load_from_disk(_path).with_format("torch")
    LOGGER.info(f"Generating new data at: {_path}")
    LOGGER.info(f"{streaming=}")

    crop_train = dataset_name == "text8-crop"
    if mode == "train" and crop_train:
        # double block size for sub-sampling
        block_size *= 2

    if dataset_name == "wikitext103":
        dataset = datasets.load_dataset(
            "wikitext",
            name="wikitext-103-raw-v1",
            cache_dir=cache_dir,
            revision=revision,
        )
    elif dataset_name == "wikitext2":
        dataset = datasets.load_dataset(
            "wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir, revision=revision
        )
    elif dataset_name == "ptb":
        dataset = datasets.load_dataset(
            "ptb_text_only", cache_dir=cache_dir, revision=revision
        )
    elif dataset_name == "lambada":
        dataset = get_lambada_test_dataset()
    elif dataset_name == "text8":
        assert wrap
        assert revision is None
        dataset = get_text8_dataset(cache_dir, max_seq_length=block_size)
    elif dataset_name == "tiny_shakespeare":
        assert wrap
        assert revision is None
        dataset = get_tiny_shakespeare_dataset(cache_dir, max_seq_length=block_size)
    elif dataset_name == WIKIPEDIA_EN_CLEAN_DATASET_NAME:
        assert wrap
        assert revision is None
        dataset = get_wikipedia_en_clean_dataset(cache_dir)
    elif dataset_name == "text8-crop":
        assert revision is None
        dataset = get_text8_dataset(
            cache_dir, max_seq_length=block_size, crop_train=True
        )
    elif dataset_name == "openwebtext-train":
        dataset = datasets.load_dataset(
            "openwebtext",
            split="train[:-100000]",
            cache_dir=cache_dir,
            revision=revision,
            streaming=False,
            trust_remote_code=True,
            download_config=datasets.DownloadConfig(num_proc=1)
        )
    elif dataset_name == "openwebtext-valid":
        dataset = datasets.load_dataset(
            "openwebtext",
            split="train[-100000:]",
            cache_dir=cache_dir,
            revision=revision,
            streaming=False,
            trust_remote_code=True,
            download_config=datasets.DownloadConfig(num_proc=1,)
        )
    elif dataset_name == "scientific_papers_arxiv":
        dataset = datasets.load_dataset(
            "scientific_papers",
            "arxiv",
            trust_remote_code=True,
            cache_dir=cache_dir,
            streaming=streaming,
            revision=revision,
        )
    elif dataset_name == "scientific_papers_pubmed":
        dataset = datasets.load_dataset(
            "scientific_papers",
            "pubmed",
            trust_remote_code=True,
            cache_dir=cache_dir,
            streaming=streaming,
            revision=revision,
        )
    elif dataset_name == "ag_news":
        dataset = datasets.load_dataset(
            "ag_news", cache_dir=cache_dir, streaming=streaming, revision=revision
        )
    elif dataset_name in ["lines", "circle", "triangle", "sibling"]:
        dataset, tokenizer, added_vocab = get_algorithmic_creativity_dataset(cache_dir, tokenizer, max_seq_length=block_size)
    else:
        dataset = datasets.load_dataset(
            dataset_name,
            cache_dir=cache_dir,
            streaming=streaming,
            trust_remote_code=True,
            revision=revision,
        )

    if dataset_name in ["lambada", "openwebtext-train", "openwebtext-valid"]:
        data = dataset
    else:
        data = dataset[mode]
    if article_limit is not None and not streaming and mode == "train":
        data = data.select(range(min(article_limit, len(data))))

    if dataset_name.startswith("wikitext"):
        detokenizer = wt_detokenizer
    elif dataset_name == "ptb":
        detokenizer = ptb_detokenizer
    elif dataset_name == "lm1b":
        detokenizer = lm1b_detokenizer
    elif dataset_name == "lambada":
        detokenizer = lambada_detokenizer
    elif dataset_name.startswith("scientific_papers"):
        detokenizer = scientific_papers_detokenizer
    else:
        detokenizer = None

    def _apply_detokenizer(detokenizer):
        def detok(text):
            for i, t in enumerate(text, 0):
                text[i] = detokenizer(t)
            return text

        return detok

    BOS = tokenizer.bos_token_id
    EOS = tokenizer.eos_token_id
    if BOS is None or EOS is None:
        raise ValueError(
            f"Tokenizer must define bos/eos token ids for wrapped dataset processing: {tokenizer}"
        )

    def preprocess_and_tokenize(example):
        if dataset_name == "ptb":
            text = example["sentence"]
        elif "scientific_papers" in dataset_name:
            text = example["article"]
        else:
            text = example["text"]

        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"

        if wrap:
            tokens = tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            if insert_eos:
                tokens = {"input_ids": [t + [EOS] for t in tokens["input_ids"]]}
            # Still missing BOS, but will be added in group_texts
        elif dataset_name == "lambada":
            tokens = tokenizer(
                text,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=True,
            )
        else:
            tokens = tokenizer(
                text,
                max_length=block_size,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=True,
            )
        return tokens
    
    
    def preprocess_and_tokenize_algorithmic(example):
        # You have input_text and target_text
        src = example["input_text"]
        tgt = example["target_text"]

        # Pre-tokenize them with no padding/truncation
        inputs = tokenizer(
            src,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        targets = tokenizer(
            tgt,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        token_offset = added_vocab[0] # account new tokens 

        return {
            "labels": inputs["input_ids"] - token_offset,
            "input_ids": targets["input_ids"] - token_offset,
            "attention_mask": targets["attention_mask"],
            "text": tgt,
        }


    if dataset_name in ['lines', 'circle', 'triangle', 'sibling']:
        process_fn = preprocess_and_tokenize_algorithmic
    else: 
        process_fn = preprocess_and_tokenize
    if streaming:
        tokenized_dataset = data.map(process_fn, batched=True)
    else:
        tokenized_dataset = data.map(
            process_fn,
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=True,
            desc="Tokenizing",
        )
    if dataset_name == "ptb":
        tokenized_dataset = tokenized_dataset.remove_columns("sentence")
    elif "scientific_papers" in dataset_name:
        tokenized_dataset = tokenized_dataset.remove_columns(
            ["article", "abstract", "section_names"]
        )
    elif dataset_name == "ag_news":
        tokenized_dataset = tokenized_dataset.remove_columns(["text", "label"])
    else:
        tokenized_dataset = tokenized_dataset.remove_columns("text")

    if not wrap:
        if not streaming:
            tokenized_dataset.save_to_disk(_path)
        return tokenized_dataset.with_format("torch")

    group_texts = functools.partial(
        _group_texts,
        block_size=block_size,
        bos=BOS,
        eos=EOS,
        insert_special_tokens=insert_special_tokens,
    )
    group_remove_columns = tokenized_dataset.column_names
    if streaming:
        chunked_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            remove_columns=group_remove_columns,
        )
    else:
        print("\n\n\n mapping dataset")
        chunked_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            num_proc=num_proc,
            batch_size=8,
            remove_columns=group_remove_columns,
            load_from_cache_file=True,
            desc="Grouping",
        )
        if not os.path.exists(_path):
            chunked_dataset.save_to_disk(_path)
        else:
            print("\n\n\n loading dataset")
            chunked_dataset = datasets.load_from_disk(_path)
    chunked_dataset = chunked_dataset.with_format("torch")
    return chunked_dataset


class VocabSizeTokenizerWrapper:
    """Compatibility wrapper used by the published s-flm checkpoints."""

    def __init__(self, tokenizer):
        object.__setattr__(self, "_tokenizer", tokenizer)

    def _wrapped(self):
        return object.__getattribute__(self, "_tokenizer")

    @property
    def vocab_size(self):
        return len(self._wrapped())

    def __len__(self):
        return len(self._wrapped())

    def __call__(self, *args, **kwargs):
        return self._wrapped()(*args, **kwargs)

    def __getattr__(self, name):
        if name == "_tokenizer":
            raise AttributeError(name)
        return getattr(self._wrapped(), name)

    def __setattr__(self, name, value):
        if name == "_tokenizer":
            object.__setattr__(self, name, value)
        else:
            setattr(self._wrapped(), name, value)

    def __repr__(self):
        return f"Wrapped<{self._wrapped()}>"


def get_tokenizer(config):
    if config.data.tokenizer_name_or_path == "hubpaths":
        from tasks.hubpaths.data import get_tokenizer as get_hubpaths_tokenizer

        tokenizer = get_hubpaths_tokenizer(config)
    elif config.data.tokenizer_name_or_path == "sudoku":
        from tasks.sudoku import SudokuTokenizer

        tokenizer = SudokuTokenizer()
    elif config.data.tokenizer_name_or_path in {"sudoku_extreme", "sudoku_shah", "maze_unique", "zebra"}:
        from tasks.reasoning import get_reasoning_tokenizer
        tokenizer = get_reasoning_tokenizer(config.data.tokenizer_name_or_path)
    elif config.data.tokenizer_name_or_path == "text8":
        tokenizer = Text8Tokenizer()
    elif config.data.tokenizer_name_or_path == "bert-base-uncased":
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.data.tokenizer_name_or_path
        )

    if isinstance(tokenizer, transformers.GPT2TokenizerFast) or isinstance(
        tokenizer, transformers.GPT2Tokenizer
    ):
        tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
            (tokenizer.bos_token, tokenizer.bos_token_id),
            (tokenizer.eos_token, tokenizer.eos_token_id),
        )

    # For wrapped batches:
    #  [BOS] sent1 [EOS] sent2-fragment [EOS]
    #  [BOS] sent2-fragment [EOS] sent3 [EOS]
    if tokenizer.bos_token is None:
        if tokenizer.cls_token is None:
            raise AttributeError(
                "Tokenizer must have a bos_token or " f"cls_token: {tokenizer}"
            )
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.eos_token is None:
        if tokenizer.sep_token is None:
            raise AttributeError(
                "Tokenizer must have a eos_token " f"or sep_token: {tokenizer}"
            )
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if config.data.tokenizer_name_or_path == "HuggingFaceTB/SmolLM-135M":
        tokenizer = VocabSizeTokenizerWrapper(tokenizer)
    return tokenizer


def get_dataloaders(
    config, tokenizer, skip_train=False, skip_valid=False, valid_seed=None
):
    skip_valid = skip_valid or config.data.get("full_training", False)
    # A CPU process is a single data-parallel worker for batch-size purposes.
    num_gpus = max(1, torch.cuda.device_count())
    if config.trainer.accumulate_grad_batches > 1:
        # assert config.loader.global_batch_size == (
        #     config.loader.batch_size
        #     * config.trainer.num_nodes
        #     * num_gpus
        #     * config.trainer.accumulate_grad_batches
        # )
        if (
            config.loader.global_batch_size
            % (num_gpus * config.trainer.accumulate_grad_batches)
            != 0
        ):
            raise ValueError(
                f"Train Batch Size {config.training.batch_size}"
                f"not divisible by {num_gpus} gpus with accumulation "
                f"{config.trainer.accumulate_grad_batches}."
            )
    if (
        config.loader.global_batch_size
        % (num_gpus * config.trainer.accumulate_grad_batches)
        != 0
    ):
        raise ValueError(
            f"Train Batch Size {config.training.batch_size}"
            f"not divisible by {num_gpus} gpus with accumulation "
            f"{config.trainer.accumulate_grad_batches}."
        )
    if config.loader.eval_global_batch_size % num_gpus != 0:
        raise ValueError(
            f"Eval Batch Size for {config.eval.batch_size} "
            f"not divisible by {num_gpus}."
        )
    if skip_train:
        train_set = None
    else:
        train_article_limit = None
        if (
            config.data.train == WIKIPEDIA_EN_CLEAN_DATASET_NAME
            and hasattr(config.data, "train_cutoff_manifest")
        ):
            train_article_limit = get_article_limit_from_manifest(
                config.data.train_cutoff_manifest
            )
        if is_task_dataset(config.data.train):
            train_set = get_task_dataset(config, tokenizer, "train")
        else:
            train_set = get_dataset(
                config.data.train,
                tokenizer,
                mode="train",
                wrap=config.data.wrap,
                insert_eos=True
                if not hasattr(config.data, "insert_train_eos")
                else config.data.insert_train_eos,
                insert_special_tokens=True
                if not hasattr(config.data, "insert_train_special")
                else config.data.insert_train_special,
                cache_dir=config.data.cache_dir,
                block_size=config.model.length,
                streaming=config.data.streaming,
                revision=config.data.get("train_revision", None),
                article_limit=train_article_limit,
            )

    if config.data.valid in ["text8", "lm1b", "ag_news"]:
        validation_split = "test"
    else:
        validation_split = "validation"
    if skip_valid:
        valid_set = None
    else:
        if is_task_dataset(config.data.valid):
            valid_set = get_task_dataset(config, tokenizer, "validation")
        else:
            valid_set = get_dataset(
                config.data.valid,
                tokenizer,
                wrap=config.data.wrap,
                insert_eos=True
                if not hasattr(config.data, "insert_valid_eos")
                else config.data.insert_valid_eos,
                insert_special_tokens=True
                if not hasattr(config.data, "insert_valid_special")
                else config.data.insert_valid_special,
                mode=validation_split,
                cache_dir=config.data.cache_dir,
                block_size=config.model.length,
                streaming=config.data.streaming,
                revision=config.data.get("valid_revision", None),
            )

    if skip_train:
        train_loader = None
    else:
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config.loader.batch_size,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=not config.data.streaming,
            drop_last=True,
            persistent_workers=config.loader.num_workers > 0,
            multiprocessing_context=(
                config.loader.get("multiprocessing_context", "spawn")
                if config.loader.num_workers > 0 else None
            ),
        )
        train_loader.tokenizer = tokenizer
    if skip_valid:
        valid_loader = None
    else:
        if valid_seed is None:
            shuffle_valid = False
            generator = None
        else:
            shuffle_valid = True
            generator = torch.Generator().manual_seed(valid_seed)
        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=config.loader.eval_batch_size,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=shuffle_valid,
            generator=generator,
            drop_last=True,
            persistent_workers=config.loader.num_workers > 0,
            multiprocessing_context=(
                config.loader.get("multiprocessing_context", "spawn")
                if config.loader.num_workers > 0 else None
            ),
        )
        # Will be used in generative perplexity calculation
        valid_loader.tokenizer = tokenizer

    return train_loader, valid_loader


# Samplers adapted from: https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py


class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):
    def __init__(self, *args, generator=None, **kwargs):
        # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
        # which should be reproducible if pl.seed_everything was called beforehand.
        # This means that changing the seed of the experiment will also change the
        # sampling order.
        if generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator().manual_seed(seed)
        kwargs.pop("shuffle", None)
        super().__init__(*args, generator=generator, **kwargs)
        self.state = self.generator.get_state()
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {"random_state": self.state, "counter": self.counter}

    def load_state_dict(self, state_dict):
        self.generator.set_state(state_dict.get("random_state"))
        self.counter = state_dict["counter"]
        # self.start_counter = self.counter
        self.restarting = True

    # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
    # epoch, and subsequent epoch will have very few batches.

    def __iter__(self) -> typing.Iterator[int]:
        n = len(self.data_source)

        self.state = self.generator.get_state()
        indices = torch.randperm(n, generator=self.generator).tolist()

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter :]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0


class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
        self.restarting = False

    def state_dict(self):
        return {"epoch": self.epoch, "counter": self.counter}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        self.counter = state_dict["counter"]
        self.restarting = True

    # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
    # epoch, and subsequent epoch will have very few batches.
    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        if not self.restarting:
            self.counter = 0
        else:
            indices = indices[self.counter :]
            self.restarting = False

        for index in indices:
            self.counter += 1
            yield index

        self.counter = 0

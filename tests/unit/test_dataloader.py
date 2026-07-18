from types import SimpleNamespace

import pytest
import torch

import dataloader


class Namespace(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def _config(*, global_batch=4, eval_global_batch=4, workers=0):
    return Namespace(
        trainer=Namespace(accumulate_grad_batches=2),
        loader=Namespace(
            global_batch_size=global_batch,
            eval_global_batch_size=eval_global_batch,
            batch_size=2,
            eval_batch_size=2,
            num_workers=workers,
            pin_memory=False,
        ),
        training=Namespace(batch_size=global_batch),
        eval=Namespace(batch_size=eval_global_batch),
        model=Namespace(length=4),
        data=Namespace(
            train="synthetic",
            valid="synthetic",
            wrap=False,
            cache_dir="unused",
            streaming=False,
        ),
    )


def test_text8_tokenizer_round_trip_and_unknown_token():
    tokenizer = dataloader.Text8Tokenizer()
    ids = tokenizer.convert_tokens_to_ids(list("abc "))

    assert tokenizer.convert_ids_to_tokens(ids) == list("abc ")
    assert tokenizer.convert_tokens_to_ids("!") == tokenizer.unk_token_id
    assert tokenizer.convert_tokens_to_string(list("abc ")) == "abc "
    assert tokenizer.vocab_size == 35


def test_tokenizer_cache_fingerprint_is_stable_and_vocab_sensitive():
    first = dataloader.Text8Tokenizer()
    second = dataloader.Text8Tokenizer()
    original = dataloader._get_tokenizer_cache_fingerprint(first)

    assert original == dataloader._get_tokenizer_cache_fingerprint(second)
    second.add_tokens(["!"])
    assert original != dataloader._get_tokenizer_cache_fingerprint(second)


def test_get_tokenizer_uses_cls_sep_and_adds_padding(monkeypatch):
    tokenizer = SimpleNamespace(
        bos_token=None,
        bos_token_id=None,
        cls_token="<cls>",
        cls_token_id=10,
        eos_token=None,
        eos_token_id=None,
        sep_token="<sep>",
        sep_token_id=11,
        pad_token=None,
        add_special_tokens=lambda values: setattr(tokenizer, "pad_token", values["pad_token"]),
    )
    monkeypatch.setattr(dataloader.transformers.AutoTokenizer, "from_pretrained", lambda _name: tokenizer)
    config = Namespace(data=Namespace(tokenizer_name_or_path="offline-tokenizer"))

    result = dataloader.get_tokenizer(config)

    assert result.bos_token == "<cls>"
    assert result.eos_token == "<sep>"
    assert result.pad_token == "[PAD]"


def test_get_tokenizer_rejects_missing_bos_fallback(monkeypatch):
    tokenizer = SimpleNamespace(bos_token=None, cls_token=None)
    monkeypatch.setattr(dataloader.transformers.AutoTokenizer, "from_pretrained", lambda _name: tokenizer)
    config = Namespace(data=Namespace(tokenizer_name_or_path="offline-tokenizer"))
    with pytest.raises(AttributeError, match="bos_token"):
        dataloader.get_tokenizer(config)


def test_skip_train_and_valid_avoids_dataset_access_on_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(dataloader, "get_dataset", lambda *_args, **_kwargs: pytest.fail("dataset accessed"))

    train, valid = dataloader.get_dataloaders(_config(), object(), skip_train=True, skip_valid=True)

    assert train is None
    assert valid is None


def test_batch_size_divisibility_validation(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="not divisible"):
        dataloader.get_dataloaders(
            _config(global_batch=6, eval_global_batch=4), object(), skip_train=True, skip_valid=True
        )
    with pytest.raises(ValueError, match="Eval Batch Size"):
        dataloader.get_dataloaders(
            _config(global_batch=8, eval_global_batch=3), object(), skip_train=True, skip_valid=True
        )


def test_validation_seed_is_deterministic(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    dataset = [{"input_ids": torch.tensor([i]), "attention_mask": torch.tensor([1])} for i in range(8)]
    monkeypatch.setattr(dataloader, "get_dataset", lambda *_args, **_kwargs: dataset)
    config = _config()

    _, first = dataloader.get_dataloaders(config, object(), skip_train=True, valid_seed=123)
    _, second = dataloader.get_dataloaders(config, object(), skip_train=True, valid_seed=123)
    _, third = dataloader.get_dataloaders(config, object(), skip_train=True, valid_seed=456)
    first_order = torch.cat([batch["input_ids"].flatten() for batch in first]).tolist()
    second_order = torch.cat([batch["input_ids"].flatten() for batch in second]).tolist()
    third_order = torch.cat([batch["input_ids"].flatten() for batch in third]).tolist()

    assert first_order == second_order
    assert first_order != third_order


def test_train_loader_supports_zero_workers(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    dataset = [{"input_ids": torch.tensor([i]), "attention_mask": torch.tensor([1])} for i in range(4)]
    monkeypatch.setattr(dataloader, "get_dataset", lambda *_args, **_kwargs: dataset)

    train, _ = dataloader.get_dataloaders(_config(workers=0), object(), skip_valid=True)

    assert len(list(train)) == 2


def test_random_fault_tolerant_sampler_resumes_exact_remaining_order():
    dataset = list(range(10))
    sampler = dataloader.RandomFaultTolerantSampler(
        dataset, generator=torch.Generator().manual_seed(42)
    )
    iterator = iter(sampler)
    consumed = [next(iterator) for _ in range(4)]
    state = sampler.state_dict()
    expected_remaining = list(iterator)

    resumed = dataloader.RandomFaultTolerantSampler(
        dataset, generator=torch.Generator().manual_seed(999)
    )
    resumed.load_state_dict(state)

    assert len(consumed) == 4
    assert list(resumed) == expected_remaining


def test_distributed_sampler_partitions_without_overlap():
    dataset = list(range(8))
    rank0 = dataloader.FaultTolerantDistributedSampler(
        dataset, num_replicas=2, rank=0, shuffle=False
    )
    rank1 = dataloader.FaultTolerantDistributedSampler(
        dataset, num_replicas=2, rank=1, shuffle=False
    )

    left, right = list(rank0), list(rank1)
    assert left == [0, 2, 4, 6]
    assert right == [1, 3, 5, 7]
    assert set(left).isdisjoint(right)


def test_distributed_sampler_resume_and_epoch_are_deterministic():
    dataset = list(range(12))
    sampler = dataloader.FaultTolerantDistributedSampler(
        dataset, num_replicas=2, rank=1, shuffle=True, seed=7
    )
    sampler.set_epoch(3)
    iterator = iter(sampler)
    next(iterator)
    next(iterator)
    state = sampler.state_dict()
    expected = list(iterator)

    resumed = dataloader.FaultTolerantDistributedSampler(
        dataset, num_replicas=2, rank=1, shuffle=True, seed=7
    )
    resumed.load_state_dict(state)
    assert list(resumed) == expected


def test_group_texts_inserts_boundaries_and_drops_remainder():
    grouped = dataloader._group_texts(
        {"input_ids": [[10, 11, 12], [13, 14, 15, 16]]},
        block_size=5,
        bos=1,
        eos=2,
        insert_special_tokens=True,
    )
    assert grouped["input_ids"] == [
        [1, 10, 11, 12, 2],
        [1, 13, 14, 15, 2],
    ]
    assert all(torch.equal(mask, torch.ones(5)) for mask in grouped["attention_mask"])


def test_group_texts_without_special_tokens():
    grouped = dataloader._group_texts(
        {"input_ids": [[1, 2, 3], [4, 5]]},
        block_size=2,
        bos=98,
        eos=99,
        insert_special_tokens=False,
    )
    assert grouped["input_ids"] == [[1, 2], [3, 4]]
    assert len(grouped["attention_mask"]) == 2


def test_group_texts_short_input_returns_empty_lists():
    grouped = dataloader._group_texts(
        {"input_ids": [[4]]}, block_size=5, bos=1, eos=2
    )
    assert grouped == {"input_ids": [], "attention_mask": []}


def test_group_texts_rejects_block_too_small_for_boundaries():
    with pytest.raises(ValueError, match="block_size"):
        dataloader._group_texts(
            {"input_ids": [[1, 2]]}, block_size=2, bos=3, eos=4
        )


def test_text8_like_normalization():
    assert dataloader.normalize_text8_like(" HéLLo,   WORLD! 123 ") == "h llo world"
    assert dataloader.normalize_text8_like("123 !!!") == ""


def test_wikipedia_cache_paths_are_scoped_to_cache_directory(tmp_path):
    paths = dataloader._get_wikipedia_en_clean_paths(str(tmp_path))
    dataset_dir = tmp_path / dataloader.WIKIPEDIA_EN_CLEAN_DATASET_NAME
    assert paths == {
        "dataset_cache_dir": str(dataset_dir),
        "split_dataset_dir": str(dataset_dir / "article_splits"),
        "metadata_path": str(dataset_dir / "prep_metadata.json"),
    }


def test_article_limit_manifest_uses_fsspec(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"article_count_cutoff": 37}')
    assert dataloader.get_article_limit_from_manifest(str(manifest)) == 37


@pytest.mark.parametrize(
    ("function", "raw", "expected"),
    [
        (dataloader.ptb_detokenizer, "it 's N \\/ ok", "it's 1 / ok"),
        (dataloader.lm1b_detokenizer, "http : / / host ? ", "http://host? "),
        (dataloader.lambada_detokenizer, "  “hello”  ", '\n"hello"'),
    ],
)
def test_detokenizers(function, raw, expected):
    assert function(raw) == expected

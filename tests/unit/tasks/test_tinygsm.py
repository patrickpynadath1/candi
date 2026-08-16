from types import SimpleNamespace

import datasets
import pytest

from tasks.tinygsm import get_tinygsm_dataset


class Tokenizer:
    bos_token_id = 10
    eos_token_id = 11
    pad_token_id = 12

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return SimpleNamespace(input_ids=[ord(character) % 10 for character in text])


def _config(tmp_path, **data_overrides):
    values = dict(
        tokenizer_name_or_path="fake/tokenizer",
        cache_dir=str(tmp_path),
        separator="\n",
        train_on_prompt=False,
        train_on_pad=False,
        filter_too_long=False,
        max_examples=None,
        wrap=False,
        val_ratio=0.5,
        val_seed=1,
    )
    values.update(data_overrides)
    return SimpleNamespace(
        model=SimpleNamespace(length=20), data=SimpleNamespace(**values)
    )


def test_tinygsm_padding_and_answer_mask(monkeypatch, tmp_path):
    raw = datasets.Dataset.from_dict(
        {"question": ["abc", "de"], "code": ["return 4", "return 5"]}
    )
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: raw)
    config = _config(tmp_path)
    dataset = get_tinygsm_dataset(config, Tokenizer())
    row = dataset["train"][0]
    assert len(row["input_ids"]) == 20
    mask = row["attention_mask"].tolist()
    assert mask[0] == 0
    assert 1 in mask
    assert mask[-1] == 0


def test_tinygsm_truncation_keeps_eos_and_never_trains_on_prompt(monkeypatch, tmp_path):
    raw = datasets.Dataset.from_dict(
        {
            "question": ["q", "r"],
            "code": ["a" * 40, "b" * 40],
        }
    )
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: raw)
    config = _config(tmp_path)
    config.model.length = 12

    dataset = get_tinygsm_dataset(config, Tokenizer())
    for split in dataset.values():
        row = split[0]
        ids = row["input_ids"].tolist()
        mask = row["attention_mask"].tolist()
        assert ids[-1] == Tokenizer.eos_token_id
        assert mask[:3] == [0, 0, 0]
        assert all(mask[3:])


def test_tinygsm_rejects_filtering_and_wrapping_together(monkeypatch, tmp_path):
    raw = datasets.Dataset.from_dict({"question": ["a", "b"], "code": ["c", "d"]})
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: raw)
    with pytest.raises(ValueError, match="cannot both"):
        get_tinygsm_dataset(
            _config(tmp_path, filter_too_long=True, wrap=True), Tokenizer()
        )

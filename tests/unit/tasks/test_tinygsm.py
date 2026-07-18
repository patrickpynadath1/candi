from types import SimpleNamespace

import datasets

from tasks.tinygsm import get_tinygsm_dataset


class Tokenizer:
    bos_token_id = 10
    eos_token_id = 11
    pad_token_id = 12

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return SimpleNamespace(input_ids=[ord(character) % 10 for character in text])


def test_tinygsm_padding_and_answer_mask(monkeypatch, tmp_path):
    raw = datasets.Dataset.from_dict(
        {"question": ["abc", "de"], "code": ["return 4", "return 5"]}
    )
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: raw)
    config = SimpleNamespace(
        model=SimpleNamespace(length=20),
        data=SimpleNamespace(
            tokenizer_name_or_path="fake/tokenizer",
            cache_dir=str(tmp_path),
            separator="\n",
            train_on_prompt=False,
            train_on_pad=False,
            filter_too_long=False,
            wrap=False,
            val_ratio=0.5,
            val_seed=1,
        ),
    )
    dataset = get_tinygsm_dataset(config, Tokenizer())
    row = dataset["train"][0]
    assert len(row["input_ids"]) == 20
    mask = row["attention_mask"].tolist()
    assert mask[0] == 0
    assert 1 in mask
    assert mask[-1] == 0

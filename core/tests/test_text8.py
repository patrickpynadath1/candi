"""Text8 token compatibility, count semantics, held-out separation, and execution."""

from dataclasses import asdict
import json

from hydra.utils import instantiate
import numpy as np
import torch

from core.data.text8 import Text8Tokenizer, pack_text
from core.tasks.metrics.text import (
    WordAccumulator,
    WordCounts,
    count_words,
    complete_words,
)
from core.tests.test_mdlm import config
from core.training.runner import run


def settings_for(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    for name, sentence in (
        ("train", " cat elephant a dog "),
        ("validation", " cat zebra elephant "),
        ("test", " tiger dog elephant "),
    ):
        (root / f"text8.{name}.txt").write_text(sentence * 40)
    return config(
        "task=text8",
        "task.length=32",
        "task.num_eval_samples=4",
        f"data.raw_dir={root}",
        f"data.cache_dir={tmp_path / 'cache'}",
        "data.limit_train=8",
        "data.limit_validation=4",
        "data.limit_test=4",
    )


def test_counts_distinguish_occurrences_unique_and_min_length():
    words = "cat cat elephant bad a".split()
    vocabulary = frozenset({"cat", "elephant", "a"})
    assert count_words(words, vocabulary) == WordCounts(5, 4, 4, 3)
    assert count_words(words, vocabulary, 3) == WordCounts(4, 3, 3, 2)
    assert count_words(words, vocabulary, 4) == WordCounts(1, 1, 1, 1)
    aggregate = WordAccumulator(vocabulary)
    aggregate.update(words)
    aggregate.update(["cat", "cat"])
    result = aggregate.compute()
    assert result.counts == WordCounts(7, 6, 4, 3)
    assert result.mean_per_sample.unique == 2.5
    assert result.rates.valid == 6 / 7
    assert result.rates.unique_valid == 3 / 4
    empty = WordAccumulator(vocabulary, 100)
    empty.update(words)
    assert empty.compute().counts == WordCounts(0, 0, 0, 0)
    assert empty.compute().rates.valid is None


def test_boundary_words_and_specials_do_not_merge():
    assert complete_words("fragment cat dog frag") == ["cat", "dog"]
    assert complete_words(" cat dog ") == ["cat", "dog"]
    assert complete_words("fragment") == []
    tokenizer = Text8Tokenizer()
    tokens = tokenizer.encode(" cat ") + [3] + tokenizer.encode(" dog ")
    assert tokenizer.word_segments(tokens) == [" cat ", " dog "]
    assert [
        w
        for s in tokenizer.word_segments(
            tokenizer.encode("ca") + [3] + tokenizer.encode("t")
        )
        for w in complete_words(s)
    ] == []
    assert tokenizer.encode("az ") == [8, 33, 34]
    assert tokenizer.mask_token_id == 4 and len(tokenizer) == 35


def test_packing_preserves_legacy_eos_grouping_and_wrapping():
    raw = b" cat dog" * 9
    rows = pack_text(raw, 8)
    source = [
        Text8Tokenizer().encode(raw[i : i + 8].decode()) + [3] for i in range(0, 72, 8)
    ]
    expected = []
    for start in range(0, len(source), 8):
        flat = sum(source[start : start + 8], [])
        expected.extend(
            [[2] + flat[i : i + 6] + [3] for i in range(0, len(flat) // 6 * 6, 6)]
        )
    np.testing.assert_array_equal(rows, expected)


def test_task_reference_uses_training_vocabulary_and_unconditional_inputs(tmp_path):
    settings = settings_for(tmp_path)
    task = instantiate(settings.task)
    task.prepare_data()
    metric = task.create_metric_accumulator("validation")
    assert "zebra" not in task._load_vocabulary()
    assert "elephant" in task._load_vocabulary()
    examples = [task.evaluation_dataset("validation")[0]]
    prompt, mask = task.prepare_generation(examples)
    assert not mask.any() and prompt.eq(0).all()
    batch = task.collate_training(examples)
    assert batch["loss_mask"].all() and batch["corruption_mask"].all()
    assert not batch["padding_mask"].any()
    metric.update(examples[0]["input_ids"][None], examples, [0])
    result = metric.compute()
    assert result.generated == result.held_out
    assert result.vocabulary_source == "train"
    assert result.reference_split == "validation"
    assert asdict(result)["records"][0]["generated"]["words"]
    model = instantiate(settings.algorithm)
    assert model.process.mask_index == model.predictor.mask_index == 4
    assert model.model_logits(prompt, torch.ones(1)).shape == (1, 32, 35)


def test_text8_trains_and_evaluates_both_samplers(tmp_path):
    settings = settings_for(tmp_path)
    settings.callback.every_n_steps = 2
    settings.output_dir = str(tmp_path / "train")
    settings.trainer.enable_progress_bar = False
    run(settings)
    assert (tmp_path / "train/task_callback/step_00000002.json").exists()
    settings.mode = "evaluate"
    settings.evaluation.checkpoint = str(tmp_path / "train/checkpoints/last.ckpt")
    for sampler in ("entropy_bounded", "ancestral"):
        settings.sampler = config(f"sampler={sampler}").sampler
        if sampler == "ancestral":
            settings.sampler.steps = 2
        settings.output_dir = str(tmp_path / sampler)
        run(settings)
        result = json.loads((tmp_path / sampler / "results.json").read_text())
        assert result["num_total"] == 2
        assert result["primary_metric"]["name"] == "valid_word_rate"
        assert len(result["records"]) == 2
        assert result["vocabulary_source"] == "train"
        assert result["held_out"]["words"]["counts"]["total"] > 0
        assert result["generated"]["words"]["counts"]["unique"] <= sum(
            r["generated"]["words"]["unique"] for r in result["records"]
        )

"""MAUVE: estimator properties, featuriser batching, caching, TinyStories and sweep wiring."""

import copy
import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("sklearn")

from core.evaluation.mauve import (  # noqa: E402
    MauveFeaturizer, MauveScorer, auto_buckets, mauve_from_features,
)
from core.sweeps import nested_tuning  # noqa: E402
from core.tests.helpers import ROOT  # noqa: E402
from core.tests.test_sol import CharTokenizer  # noqa: E402

EVALUATION = ROOT / "experiments/rerun/evaluation"


def cloud(n, seed, shift=0.0, dimension=16):
    rng = np.random.default_rng(seed)
    center = np.full(dimension, 3.0)
    center[0] += shift
    return (center + rng.standard_normal((n, dimension))).astype(np.float32)


class FakeFeaturizer:
    """Deterministic per-text features; counts the texts it featurises."""

    provenance = {"model": "fake", "revision": "0"}

    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += len(texts)
        rows = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            rows.append(np.random.default_rng(seed).standard_normal(8) + 3)
        return np.asarray(rows, dtype=np.float32)


def test_identical_samples_score_one_and_separated_clouds_zero():
    p = cloud(300, 0)
    assert mauve_from_features(p, p.copy(), num_buckets=30)[0] == pytest.approx(1.0)
    assert mauve_from_features(p, cloud(300, 1), num_buckets=30)[0] > 0.9  # same law
    assert mauve_from_features(p, cloud(300, 1, shift=30.0), num_buckets=30)[0] < 0.02


def test_more_noise_lowers_mauve_monotonically():
    p, q, noise = cloud(400, 0), cloud(400, 1), np.random.default_rng(2).standard_normal((400, 16))
    values = [mauve_from_features(p, q + s * noise, num_buckets=40)[0] for s in (0, 1, 2, 4, 8)]
    assert all(a > b for a, b in zip(values, values[1:])), values


def test_order_invariant_and_deterministic_with_a_fixed_seed():
    p, q = cloud(200, 0), cloud(150, 1, shift=1.0)
    value = mauve_from_features(p, q, num_buckets=20, seed=7)[0]
    shuffle = np.random.default_rng(3)
    assert mauve_from_features(p[shuffle.permutation(200)], q[shuffle.permutation(150)],
                               num_buckets=20, seed=7)[0] == value
    assert mauve_from_features(p, q, num_buckets=20, seed=7)[0] == value
    assert mauve_from_features(p, q)[1] == auto_buckets(200, 150) == 15
    with pytest.raises(ValueError, match="num_buckets"):
        mauve_from_features(p, q, num_buckets=1)


def test_featurizer_takes_the_last_token_independent_of_batching_and_keeps_the_end():
    from transformers import GPT2Config, GPT2Model

    torch.manual_seed(0)
    model = GPT2Model(GPT2Config(vocab_size=256, n_positions=12, n_embd=16, n_layer=1, n_head=2))
    texts = ["a cat", "the dog ran home", "x", "a long story that overflows"]
    batched = MauveFeaturizer("tiny", device="cpu", model=model, tokenizer=CharTokenizer())(texts)
    single = MauveFeaturizer("tiny", device="cpu", model=model, tokenizer=CharTokenizer(),
                             batch_size=1)
    for text, row in zip(texts, batched):
        np.testing.assert_allclose(single([text])[0], row, atol=1e-5)
    with torch.no_grad():  # over 12 tokens: the last 12 are kept, ending on the final token
        ids = torch.tensor([list(texts[3].encode())[-12:]])
        expected = model(input_ids=ids).last_hidden_state[0, -1].numpy()
    np.testing.assert_allclose(batched[3], expected, atol=1e-5)


def test_scorer_counts_each_reference_once_and_caches_them(tmp_path):
    prompts = [f"Prompt {i}. " for i in range(40)]
    references = [f"story {i} ends well." for i in range(40)]
    featurizer = FakeFeaturizer()
    scorer = MauveScorer(featurizer, num_buckets=5, cache_dir=tmp_path)
    result = scorer.score(prompts * 2, references * 2, references * 2, list(range(40)) * 2)
    assert result["mauve"] == pytest.approx(1.0)  # generations equal to the references
    signature = result["mauve_signature"]
    assert (signature["n_references"], signature["n_generated"], signature["num_buckets_used"]) \
        == (40, 80, 5)
    assert featurizer.calls == 120
    other = [f"something else {i}" for i in range(40)]
    worse = scorer.score(prompts, other, references, range(40))
    assert worse["mauve"] < result["mauve"] and featurizer.calls == 160  # references cached
    interval = MauveScorer(featurizer, num_buckets=5, cache_dir=tmp_path, bootstrap=20).score(
        prompts, other, references, range(40))["mauve_interval"]
    assert interval["low"] <= interval["high"] and interval["replicates"] == 20


def test_tinystories_wires_mauve_as_a_maximised_primary_and_a_reported_cross_check(tmp_path):
    from core.tasks.tinystories import TinyStoriesTask
    from core.tests.test_tinystories import fixture_data

    with pytest.raises(ValueError, match="task.mauve"):
        TinyStoriesTask(fixture_data(tmp_path), length=40, prompt_bytes=4, primary_metric="mauve")
    task = TinyStoriesTask(
        fixture_data(tmp_path), length=40, prompt_bytes=4, primary_metric="mauve",
        mauve=MauveScorer(FakeFeaturizer(), num_buckets=2, cache_dir=None),
        mauve_gpt2_large=MauveScorer(FakeFeaturizer(), num_buckets=2, cache_dir=None),
    )
    assert (task.primary_metric.name, task.primary_metric.mode) == ("mauve", "max")
    dataset = task.validation_dataset()
    examples = [dataset[i] for i in range(len(dataset))]
    metric = task.create_metric_accumulator("validation")
    metric.update([e["input_ids"] for e in examples], examples, range(len(examples)))
    result = metric.compute()
    assert result.mauve == pytest.approx(1.0) and result.mauve_gpt2_large == pytest.approx(1.0)
    assert result.mauve_gpt2_large_signature["num_buckets_used"] == 2
    # Records saved as JSON score the same when reloaded (the pooled-seed path).
    saved = json.loads(json.dumps(asdict(result)))["records"]
    again = task.create_metric_accumulator("validation")
    again.load_records(saved)
    assert again.compute().mauve == result.mauve


class FakeTask:
    def __init__(self, scorer):
        self.scorer = scorer

    def create_metric_accumulator(self, name):
        from core.tasks.tinystories_metrics import TinyStoriesMetrics

        return TinyStoriesMetrics(None, mauve=self.scorer)


def fake_task():
    return FakeTask(MauveScorer(FakeFeaturizer(), num_buckets=2, cache_dir=None))


def test_pooled_result_scores_all_seeds_as_one_sample():
    def record(index, text):
        return {"id": str(index), "index": index, "prompt": "p", "continuation": text,
                "reference": f"r{index}", "generated_text": "p" + text, "token_count": 1,
                "word_count": 1, "eos_present": True, "invalid_special_count": 0,
                "invalid_utf8": False, "repeated_trigram_fraction": 0.0, "nfe": 4}

    seeds = [{"records": [record(i, f"r{i}") for i in range(10)]},
             {"records": [record(i, f"other {i}") for i in range(10)]}]
    config = OmegaConf.create({"task": {"_target_": "core.tests.test_mauve.fake_task"}})
    pooled = nested_tuning.pooled_result(config, "validation", seeds)
    assert (pooled["mauve_signature"]["n_generated"], pooled["mauve_signature"]["n_references"]) \
        == (20, 10)
    assert "records" not in pooled and 0 < pooled["mauve"] < 1


def spec_with(evaluation):
    """A minimal valid nested-tuning case around an evaluation file."""
    return {
        "training": {"checkpoint": "x.ckpt", "space": {}, "trials": 1},
        "evaluation": {**evaluation, "nfe": 128},
        "inference": [{"name": "native", "sampler": {}, "space": {}, "trials": 4,
                       "budget_parameter": "steps"}],
    }


def test_mauve_evaluation_file_keeps_the_partitions_and_lands_every_override():
    spec = json.loads((EVALUATION / "tinystories-bpe-10000-mauve.json").read_text())
    old = json.loads((EVALUATION / "tinystories-bpe-10000.json").read_text())
    for key in ("task", "seeds", "test_seeds", "tuning_indices", "test_indices"):
        assert spec[key] == old[key]
    assert {k: v for k, v in spec["two_stage"].items() if k != "stage2_overrides"} \
        == {k: v for k, v in old["two_stage"].items() if k != "stage2_overrides"}
    assert spec["score"] == spec["test_score"] == {"kind": "metric", "metric": "mauve",
                                                   "pool_seeds": True}
    assert spec["direction"] == "maximize"
    nested_tuning.validate(spec_with(spec))
    bad = copy.deepcopy(spec)
    bad["score"]["pool_seeds"] = "yes"
    with pytest.raises(ValueError, match="pool_seeds"):
        nested_tuning.validate(spec_with(bad))
    yaml = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/task/mauve/tinystories_33m.yaml"))
    assert spec["overrides"]["task.mauve"] == {**yaml, "num_buckets": 13}

    def task_after(overrides):
        config = OmegaConf.create({"task": {"_target_": "core.tasks.tinystories.TinyStoriesTask"}})
        for key, value in overrides.items():  # as evaluation_config applies them, in order
            nested_tuning.update(config, key, value, merge=False, force_add=True)
        return OmegaConf.to_container(config.task)

    stage1 = task_after(spec["overrides"])
    assert stage1["primary_metric"] == "mauve" and stage1["mauve"]["num_buckets"] == 13
    assert "ref_w1" not in stage1 and "sol" not in stage1
    stage2 = task_after({**spec["overrides"], **spec["two_stage"]["stage2_overrides"]})
    assert stage2["mauve"] == {**stage1["mauve"], "num_buckets": 26}
    final = {**{k: v for k, v in spec.items() if k not in ("test_score", "test_overrides")}}
    settings = nested_tuning.test_settings(spec, final)
    assert settings["score"]["metric"] == "mauve"
    test = task_after(settings["overrides"])
    assert test["mauve"] == {**stage1["mauve"], "num_buckets": 205}
    assert test["sol"] == old["test_overrides"]["task.sol"]
    cross = test["mauve_gpt2_large"]
    assert (cross["featurizer"]["model_name"], cross["num_buckets"]) == ("gpt2-large", 205)
    assert test["primary_metric"] == "mauve" and "ref_w1" not in test
    # Extra-budget measurements (128 prompts x 1 seed) get the bucket count of that size.
    extra = task_after({**settings["overrides"], **spec["extra_test_overrides"]})
    assert extra["mauve"]["num_buckets"] == extra["mauve_gpt2_large"]["num_buckets"] \
        == auto_buckets(128, 128) == 13
    assert extra["sol"] == test["sol"]
    bad = copy.deepcopy(spec)
    bad["extra_test_overrides"] = {"sampler.steps": 3}
    with pytest.raises(ValueError, match="evaluation override"):
        nested_tuning.validate(spec_with(bad))

"""ref_w1: exact 1-D W1, prompt-conditioned coding rates, NLLs, floors and TinyStories wiring."""

import random
from fractions import Fraction

import numpy as np
import pytest
import torch

pytest.importorskip("zstandard")

from core.evaluation.ref_w1 import (  # noqa: E402
    CausalScorer, RateCoder, RefW1Scorer, build_dictionary, wasserstein_1,
)
from core.tests.test_sol import CharTokenizer  # noqa: E402

WORDS = ("the a little girl boy dog cat sun tree park ball happy sad big small ran saw "
         "played went home and with to her his mom said one day they were very").split()


def stories(n, seed=0, words=60):
    rng = random.Random(seed)
    return [" ".join(rng.choice(WORDS) for _ in range(words)).capitalize() + "." for _ in range(n)]


FIT = stories(400, seed=1)


def brute_force_w1(a, b):
    """W1 via quantile functions on the common grid of 1/(len(a) len(b)) steps."""
    a, b = sorted(a), sorted(b)
    grid = len(a) * len(b)
    total = Fraction(0)
    for k in range(grid):
        total += abs(Fraction(a[k // len(b)]) - Fraction(b[k // len(a)]))
    return float(total / grid)


def test_w1_zero_symmetric_and_exact_for_unequal_sizes():
    rng = np.random.default_rng(0)
    a = rng.normal(size=7).round(3)
    assert wasserstein_1(a, a.copy()) == 0.0
    for n, m in [(5, 5), (4, 9), (1, 6)]:
        x, y = rng.normal(size=n).round(3), (rng.normal(size=m) * 2 + 0.5).round(3)
        assert wasserstein_1(x, y) == pytest.approx(wasserstein_1(y, x), abs=1e-12)
        assert wasserstein_1(x, y) == pytest.approx(brute_force_w1(x, y), rel=1e-9, abs=1e-12)
    assert wasserstein_1([0, 0], [3]) == pytest.approx(3.0)  # no truncation of either list


def test_rates_fall_for_repetition_rise_for_noise_and_w1_grows_both_ways():
    coder = RateCoder(build_dictionary(FIT))
    real = stories(64, seed=2)
    rng = random.Random(3)
    repetitive = ["the dog " * 40 for _ in real]
    noise = ["".join(rng.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(360)) for _ in real]
    rates = {name: [coder.rate(t) for t in texts]
             for name, texts in [("real", real), ("repetitive", repetitive), ("noise", noise)]}
    assert np.mean(rates["repetitive"]) < np.mean(rates["real"]) < np.mean(rates["noise"])
    held_out = [coder.rate(t) for t in stories(64, seed=4)]
    close = wasserstein_1(rates["real"], held_out)
    assert wasserstein_1(rates["real"], rates["repetitive"]) > 10 * close
    assert wasserstein_1(rates["real"], rates["noise"]) > 10 * close


def test_prompt_conditioning_subtracts_the_prompt_cost():
    coder = RateCoder(build_dictionary(FIT))
    prompt, continuation = "Once upon a time a little girl saw a big dog.", " The dog ran home."
    rate = coder.conditional_rate(prompt, continuation)
    tail = len(continuation.encode())
    assert rate * tail == coder.bits(prompt + continuation) - coder.bits(prompt)
    # Repeating the prompt is nearly free once the prompt has been paid for.
    assert coder.conditional_rate(prompt, prompt) < coder.rate(prompt)
    assert coder.conditional_rate(prompt, "") == coder.conditional_rate(prompt, "\n")
    with pytest.raises(ValueError, match="fewer than"):
        build_dictionary(["short"], 64)


def tiny_lm():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=256, n_positions=64, n_embd=16, n_layer=1, n_head=2))
    return CausalScorer("tiny", device="cpu", dtype="float32", model=model,
                        tokenizer=CharTokenizer(), batch_size=2)


def test_nll_matches_a_direct_computation_and_ignores_padding():
    scorer = tiny_lm()
    prompts, continuations = ["ab", "hello there", "x"], ["cd", "", "a much longer one"]
    values = scorer.nll(prompts, continuations)
    for prompt, continuation, value in zip(prompts, continuations, values):
        head = list(prompt.encode())
        tail = list(continuation.encode()) or [255]
        ids = torch.tensor([head + tail])
        logp = torch.log_softmax(scorer.model(input_ids=ids).logits[0], -1)
        expected = -torch.stack([logp[len(head) + j - 1, t] for j, t in enumerate(tail)]).mean()
        assert value == pytest.approx(float(expected), rel=1e-5)


def scorer(tmp_path, **options):
    return RefW1Scorer(FIT, nll=options.pop("nll", None), cache_dir=tmp_path, **options)


def test_floors_are_cached_deterministic_and_order_free(tmp_path, monkeypatch):
    prompts, refs = stories(40, seed=5, words=8), stories(40, seed=6)
    generated = stories(40, seed=7)
    first = scorer(tmp_path).score(prompts, generated, refs)
    calls = []
    again = scorer(tmp_path)
    original = again.signals
    monkeypatch.setattr(again, "signals", lambda p, c: calls.append(len(p)) or original(p, c))
    order = list(range(40))
    random.Random(0).shuffle(order)
    second = again.score([prompts[i] for i in order], [generated[i] for i in order],
                         [refs[i] for i in order])
    assert calls == [40]  # only the generated side: the real side came from the cache
    for key in ("ref_w1", "ref_w1_rate", "ref_w1_rate_floor"):
        assert second[key] == pytest.approx(first[key], rel=1e-12)
    assert scorer(tmp_path / "fresh").score(prompts, refs, refs)["ref_w1"] == 0.0


def test_combined_score_with_nll_bootstrap_and_monotonicity(tmp_path):
    lm = tiny_lm()
    prompts, refs = stories(16, seed=8, words=4), stories(16, seed=9, words=12)
    result = scorer(tmp_path, nll=lm, bootstrap=20).score(prompts, ["the dog " * 6] * 16, refs)
    assert result["ref_w1"] == pytest.approx(
        result["ref_w1_rate"] / result["ref_w1_rate_floor"]
        + result["ref_w1_nll"] / result["ref_w1_nll_floor"])
    assert result["ref_w1_interval"]["low"] <= result["ref_w1_interval"]["high"]
    floors = {"rate": 0.5, "nll": 2.0}
    base = RefW1Scorer.combine({"rate": 1.0, "nll": 1.0}, floors)
    assert RefW1Scorer.combine({"rate": 1.5, "nll": 1.0}, floors) > base
    assert RefW1Scorer.combine({"rate": 1.0, "nll": 1.5}, floors) > base
    with pytest.raises(ValueError, match="sha256"):
        RefW1Scorer(FIT, dictionary_sha256="0" * 64)


def test_tinystories_records_ref_w1_from_its_fit_split(tmp_path):
    from core.tasks.tinystories import TinyStoriesTask
    from core.tests.test_tinystories import fixture_data

    settings = {"fit_indices": [0, 1], "dictionary_bytes": 32, "nll": None,
                "cache_dir": str(tmp_path / "cache")}
    task = TinyStoriesTask(fixture_data(tmp_path), length=40, prompt_bytes=4,
                           ref_w1=settings, primary_metric="ref_w1")
    assert (task.primary_metric.name, task.primary_metric.mode) == ("ref_w1", "min")
    dataset = task.validation_dataset()
    examples = [dataset[i] for i in range(len(dataset))]
    metric = task.create_metric_accumulator("validation")
    metric.update([e["input_ids"] for e in examples], examples, range(len(examples)))
    result = metric.compute()
    assert result.ref_w1 == 0.0 and result.ref_w1_rate_floor > 0
    assert result.ref_w1_signature["dictionary_bytes"] == 32
    with pytest.raises(ValueError, match="task.ref_w1"):
        TinyStoriesTask(fixture_data(tmp_path), length=40, prompt_bytes=4, primary_metric="ref_w1")

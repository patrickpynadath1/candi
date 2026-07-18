import math

import pytest

import text_metrics


def test_ngram_coverage_word_and_character_modes():
    assert text_metrics.calculate_ngram_coverage(
        ["a b c x"], ["a b c d"], n=3
    ) == pytest.approx(0.5)
    assert text_metrics.calculate_ngram_coverage(
        ["abcz"], ["abcd"], n=3, word_based=False
    ) == pytest.approx(0.5)


def test_ngram_metrics_handle_empty_and_short_inputs():
    assert text_metrics.calculate_ngram_coverage(["short"], ["short"], n=2) == 0
    assert text_metrics.unique_ngrams_per_sample(["a"], n=2) == 0
    assert text_metrics.total_unique_ngrams(["a"], n=2) == 0
    assert text_metrics.calculate_repetition_rate(["a"], n=2) == 0


def test_unique_ngram_and_repetition_math():
    text = "a b a b"
    assert text_metrics.unique_ngrams_per_sample([text], n=2) == pytest.approx(2 / 3)
    assert text_metrics.calculate_repetition_rate([text], n=2) == pytest.approx(0.5)


def test_character_distribution_and_empty_input():
    freqs, divergence = text_metrics.calculate_char_distribution_metrics(
        ["aaab"], ["aabb"]
    )
    assert freqs == {"a": 0.75, "b": 0.25}
    assert math.isfinite(divergence)
    assert text_metrics.calculate_char_distribution_metrics([], []) == ({}, 0.0)


def test_dictionary_metrics_use_supplied_training_texts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train = ["alpha beta alphabet"]
    assert text_metrics.calculate_word_dictionary_match(
        ["alphabet missing"], train, n=4
    ) == pytest.approx(0.5)
    assert text_metrics.total_valid_words(["alpha missing"], train) == pytest.approx(0.5)
    assert list(tmp_path.iterdir()) == []

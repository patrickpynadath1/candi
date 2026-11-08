"""core.analysis reads nested-tuning case trees and merges re-tuned cases."""

import json

import pytest

from core.analysis import bleu_parts, load_cases, merge_cases


def write_case(root, name, policies, selected, test):
    case = root / name
    (case / "final").mkdir(parents=True)
    spec = {"name": "c", "inference": [{"name": p} for p in policies]}
    config = {"benchmark": {"task_id": "text8", "algorithm_name": "masked"}}
    (case / "protocol.json").write_text(json.dumps({"config": config, "spec": spec}))
    for policy, score in policies.items():
        path = case / "training/candidates/abc/inference" / policy / "selected.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"score": score, "seed_scores": [score, score], "parameters": {}}))
    result = {"score": test, "seed_scores": [test], "selection": {"policy": {"name": selected}}}
    (case / "final/result.json").write_text(json.dumps(result))
    (case / "final/seed-1041").mkdir()
    records = [{"continuation": "a b c d", "reference": "a b c d e"}]
    (case / "final/seed-1041/results.json").write_text(json.dumps({"records": records}))


def test_cases_load_and_a_retuned_case_takes_the_overall_winner(tmp_path):
    write_case(tmp_path / "first", "c", {"native": 0.5, "eb": 0.7}, "eb", 0.71)
    write_case(tmp_path / "renoise", "c", {"renoise": 0.9}, "renoise", 0.88)
    cases = load_cases(tmp_path)
    assert len(cases) == 2 and cases[0].task == "text8"
    (merged,) = merge_cases(cases)
    assert list(merged.policies) == ["native", "eb", "renoise"]
    assert merged.winner() == ("renoise", ["renoise"])
    assert (merged.selected, merged.test) == ("renoise", 0.88)
    assert list(merged.records()) == [1041]


def test_bleu_without_brevity_penalty_divides_out_the_penalty():
    pytest.importorskip("sacrebleu")
    records = [{"continuation": "the cat sat on", "reference": "the cat sat on the mat"}]
    parts = bleu_parts(records)
    assert parts["bp"] < 1 and parts["len_ratio"] == pytest.approx(4 / 6)
    assert parts["bleu_nobp"] == pytest.approx(parts["bleu"] / parts["bp"])

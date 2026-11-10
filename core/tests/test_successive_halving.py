"""Successive-halving sampler tuning and several NFE budgets in nested tuning."""

import copy
import json
import math

import pytest

from core.sweeps import nested_tuning as tuning
from core.tests.test_nested_tuning import spec as base_spec

TEMPERATURES = [round(0.5 + 0.04 * i, 2) for i in range(27)]
POOL, TEST = list(range(100, 140)), list(range(200, 210))


def spec(tmp_path, nfe=4):
    tmp_path.mkdir(parents=True, exist_ok=True)
    settings = base_spec(tmp_path, fixed=True)
    settings["evaluation"].update(
        tuning_indices=POOL,
        test_indices=TEST,
        nfe=nfe,
        direction="minimize",
        score={"kind": "metric", "metric": "sol"},
        halving={
            "rungs": [4, 12, 36],
            "keep": 1 / 3,
            "finalists": 3,
            "finalist_seeds": [77],
            "finalist_overrides": {"task.sol.bootstrap": 10},
        },
    )
    settings["inference"][0].update(
        space={"algorithm.predictor.temperature": TEMPERATURES},
        trials=len(TEMPERATURES),
        initial_points=[{"algorithm.predictor.temperature": t} for t in TEMPERATURES],
    )
    return settings


def score(temperature, prompts, seeds, nfe=4):
    # Small rungs are biased towards high temperatures and the fresh-seed re-score
    # prefers the highest finalist, so a correct schedule re-ranks at every stage.
    bias = {4: 0.3, 12: 0.1, 36: 0.0}[prompts] if seeds != [77] else 0.0
    return abs(temperature - 1.0 - bias) - 2 * (seeds == [77]) * temperature + 1 / nfe


class Recorder:
    def __init__(self, fail_at=None):
        self.calls, self.fail_at = [], fail_at

    def __call__(self, stage, payload, output):
        evaluation = payload["evaluation"]
        assert "test_seeds" not in evaluation and "tuning_indices" not in evaluation
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            self.fail_at = None
            raise RuntimeError("worker lost")
        self.calls.append(copy.deepcopy(payload))
        t = payload["parameters"]["algorithm.predictor.temperature"]
        if evaluation["indices"] == TEST:
            return {"score": -1.0}
        value = score(t, len(evaluation["indices"]), evaluation["seeds"], evaluation["nfe"])
        result = {"score": value, "seed_scores": [value]}
        if evaluation["seeds"] == [77]:
            result["score_interval"] = {"low": value - 0.01, "high": value + 0.01}
        return result


def rung_calls(recorder, prompts, seeds=(41,)):
    return [
        c for c in recorder.calls
        if len(c["evaluation"]["indices"]) == prompts and c["evaluation"]["seeds"] == list(seeds)
        and c["evaluation"]["indices"] != TEST
    ]


def best(temperatures, prompts):
    return sorted(temperatures, key=lambda t: (score(t, prompts, [41]), TEMPERATURES.index(t)))


def test_halving_promotes_the_best_third_at_each_rung(tmp_path, monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(tuning, "execute", recorder)
    winner = tuning.run_case(spec(tmp_path), tmp_path / "runs", concurrency=2)
    temperature = lambda c: c["parameters"]["algorithm.predictor.temperature"]  # noqa: E731
    first = rung_calls(recorder, 4)
    assert sorted(map(temperature, first)) == TEMPERATURES  # every trial, on 4 prompts
    second = rung_calls(recorder, 12)
    assert sorted(map(temperature, second)) == sorted(best(TEMPERATURES, 4)[:9])
    third = rung_calls(recorder, 36)
    assert sorted(map(temperature, third)) == sorted(best(map(temperature, second), 12)[:3])
    finalists = rung_calls(recorder, 36, seeds=(77,))
    assert sorted(map(temperature, finalists)) == sorted(map(temperature, third))
    # Compute matches the rung: nested prefixes of the tuning pool, never the test split.
    for prompts, calls in [(4, first), (12, second), (36, third + finalists)]:
        assert all(c["evaluation"]["indices"] == POOL[:prompts] for c in calls)
    assert all(c["evaluation"]["overrides"] == {"task.sol.bootstrap": 10} for c in finalists)
    assert all("overrides" not in c["evaluation"] for c in first + second + third)
    # Selection is by the fresh-seed re-score, and its interval is recorded.
    chosen = min(map(temperature, finalists), key=lambda t: score(t, 36, [77]))
    assert winner["inference_parameters"]["algorithm.predictor.temperature"] == chosen
    assert chosen != best(map(temperature, third), 36)[0] or len(third) == 1
    assert winner["score_interval"]["low"] < winner["score"] < winner["score_interval"]["high"]
    selected = json.loads((tmp_path / "runs/example/training/candidates").glob("*/inference/native/selected.json").__next__().read_text())
    assert [len(r["scores"]) for r in selected["halving"]["rungs"]] == [27, 9, 3]
    # The test split is touched once, last, for the winner only.
    tests = [i for i, c in enumerate(recorder.calls) if c["evaluation"]["indices"] == TEST]
    assert tests == [len(recorder.calls) - 1]
    assert recorder.calls[-1]["evaluation"]["seeds"] == [1041]
    assert "halving" not in recorder.calls[-1]["evaluation"]
    assert len(recorder.calls) == 27 + 9 + 3 + 3 + 1
    count = len(recorder.calls)
    assert tuning.run_case(spec(tmp_path), tmp_path / "runs", concurrency=2) == winner
    assert len(recorder.calls) == count  # an idempotent resume


@pytest.mark.parametrize("fail_at", [5, 30, 37, 40])  # rung 0, rung 1, rung 2, finalists
def test_resume_after_interruption_reproduces_the_selection(tmp_path, monkeypatch, fail_at):
    reference = Recorder()
    monkeypatch.setattr(tuning, "execute", reference)
    expected = tuning.run_case(spec(tmp_path / "a"), tmp_path / "a/runs", concurrency=1)
    interrupted = Recorder(fail_at=fail_at)
    monkeypatch.setattr(tuning, "execute", interrupted)
    settings = spec(tmp_path / "b")
    settings["inference"][0]["on_failure"] = "stop"
    with pytest.raises(Exception, match="worker lost"):
        tuning.run_case(settings, tmp_path / "b/runs", concurrency=1)
    assert not any(c["evaluation"]["indices"] == TEST for c in interrupted.calls)
    done = len(interrupted.calls)
    resumed = tuning.run_case(settings, tmp_path / "b/runs", concurrency=1)
    assert resumed["inference_parameters"] == expected["inference_parameters"]
    assert resumed["score"] == expected["score"]
    # Nothing finished before the interruption is evaluated again.
    assert len(interrupted.calls) == len(reference.calls)
    assert done == fail_at


def test_halving_is_opt_in_and_validated(tmp_path):
    plain = base_spec(tmp_path, fixed=True)
    tuning.validate(plain)
    assert tuning.generation_cost(plain)["tuning_sequences"] == 2 * 2 * 1
    settings = spec(tmp_path)
    cost = tuning.generation_cost(settings)
    assert cost["tuning_sequences"] == 27 * 4 + 9 * 12 + 3 * 36 + 3 * 36
    for broken in (
        {"rungs": [12, 4]},
        {"rungs": [4, 64]},  # beyond the 40-prompt pool
        {"finalist_seeds": [41]},  # not fresh
        {"finalist_seeds": [1041]},
        {"finalist_overrides": {"sampler.steps": 3}},
    ):
        bad = spec(tmp_path)
        bad["evaluation"]["halving"].update(broken)
        with pytest.raises(ValueError):
            tuning.validate(bad)


def test_several_nfe_budgets_are_tuned_and_tested_separately(tmp_path, monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(tuning, "execute", recorder)
    winner = tuning.run_case(spec(tmp_path, nfe=[4, 8]), tmp_path / "runs", concurrency=2)
    assert set(winner["budgets"]) == {"4", "8"}
    assert winner["budgets"]["8"]["nfe"] == 8 and winner["nfe"] == 4
    tests = [c for c in recorder.calls if c["evaluation"]["indices"] == TEST]
    assert sorted(c["evaluation"]["nfe"] for c in tests) == [4, 8]
    root = tmp_path / "runs/example"
    assert (root / "final/result.json").exists() and (root / "final-nfe-8/result.json").exists()
    candidate = next((root / "training/candidates").iterdir())
    assert (candidate / "inference/native/selected.json").exists()
    assert (candidate / "inference-nfe-8/native/selected.json").exists()
    per_budget = len(recorder.calls) - 2
    assert per_budget == 2 * (27 + 9 + 3 + 3)
    assert {c["evaluation"]["nfe"] for c in recorder.calls} == {4, 8}
    assert math.isclose(tuning.generation_cost(spec(tmp_path, nfe=[4, 8]))["network_calls"],
                        (27 * 4 + 9 * 12 + 3 * 36 + 3 * 36 + 10) * (4 + 8))


def test_conditional_parameters_exist_only_under_their_condition(tmp_path):
    import optuna

    space = {
        "sampler.transition.schedule": ["rescale", "loop"],
        "sampler.transition.t_on": {"type": "float", "low": 0.5, "high": 1.0,
                                    "when": {"sampler.transition.schedule": "loop"}},
        "algorithm.predictor.top_p": {"type": "categorical", "choices": [0.9, 1.0]},
    }
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    seen = set()
    for _ in range(20):
        values = tuning.suggest(study.ask(), space)
        loop = values["sampler.transition.schedule"] == "loop"
        assert ("sampler.transition.t_on" in values) == loop
        assert values["algorithm.predictor.top_p"] in (0.9, 1.0)
        seen.add(loop)
    assert seen == {True, False}
    settings = base_spec(tmp_path, fixed=True)
    settings["inference"][0].update(space=space, trials=6, initial_points=[])
    tuning.validate(settings)
    settings["inference"][0]["space"] = dict(reversed(list(space.items())))
    with pytest.raises(ValueError, match="earlier parameter"):
        tuning.validate(settings)
    settings["inference"][0]["space"] = {**space, "algorithm.predictor.top_k": [0, 5]}
    with pytest.raises(ValueError, match="Not a tunable"):
        tuning.validate(settings)

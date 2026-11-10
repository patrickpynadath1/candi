"""Two-stage sampler selection, per-policy Optuna samplers, extra test budgets, NFE lists."""

import copy
import json
import math

import pytest

from core.sweeps import nested_tuning as tuning
from core.tests.test_nested_tuning import spec as base_spec

TEMPERATURES = [round(0.5 + 0.04 * i, 2) for i in range(12)]
POOL, TEST = list(range(100, 140)), list(range(200, 210))
STAGE1, STAGE2 = POOL[:8], POOL[8:24]


def spec(tmp_path, nfe=4):
    tmp_path.mkdir(parents=True, exist_ok=True)
    settings = base_spec(tmp_path, fixed=True)
    settings["evaluation"].update(
        tuning_indices=POOL,
        test_indices=TEST,
        nfe=nfe,
        direction="minimize",
        score={"kind": "metric", "metric": "sol"},
        two_stage={"stage1": 8, "stage2": 16, "finalists": 3, "stage2_seeds": [77],
                   "stage2_overrides": {"task.sol.bootstrap": 10}},
    )
    settings["inference"][0].update(
        space={"algorithm.predictor.temperature": TEMPERATURES},
        trials=len(TEMPERATURES),
        initial_points=[{"algorithm.predictor.temperature": t} for t in TEMPERATURES],
    )
    return settings


def score(temperature, indices, seeds, nfe=4):
    # Stage 1 favours temperatures near 1.2; stage 2 prefers the lowest finalist, so the
    # stage-2 winner differs from the stage-1 leader.
    if seeds == [77]:
        return abs(temperature - 1.2) + 2 * temperature + 1 / nfe
    return abs(temperature - 1.2) + 0.001 * temperature + 1 / nfe


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
        if set(evaluation["indices"]) <= set(TEST):  # the test split or its smoke subset
            return {"score": -1.0}
        value = score(t, evaluation["indices"], evaluation["seeds"], evaluation["nfe"])
        result = {"score": value, "seed_scores": [value]}
        if evaluation["seeds"] == [77]:
            result["score_interval"] = {"low": value - 0.01, "high": value + 0.01}
        return result


def temperature(call):
    return call["parameters"]["algorithm.predictor.temperature"]


def stage(recorder, indices, seeds):
    return [c for c in recorder.calls
            if c["evaluation"]["indices"] == indices and c["evaluation"]["seeds"] == seeds]


def test_two_stage_promotes_the_top_three_and_selects_by_stage_two(tmp_path, monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(tuning, "execute", recorder)
    winner = tuning.run_case(spec(tmp_path), tmp_path / "runs", concurrency=2)
    first = stage(recorder, STAGE1, [41])
    assert sorted(map(temperature, first)) == TEMPERATURES  # every trial on stage 1
    ranked = sorted(TEMPERATURES, key=lambda t: (score(t, STAGE1, [41]), TEMPERATURES.index(t)))
    second = stage(recorder, STAGE2, [77])
    assert sorted(map(temperature, second)) == sorted(ranked[:3])  # exactly the top 3
    assert not set(STAGE1) & set(STAGE2)  # disjoint prompts, fresh seed
    assert all(c["evaluation"]["overrides"] == {"task.sol.bootstrap": 10} for c in second)
    assert all("overrides" not in c["evaluation"] for c in first)
    chosen = min(ranked[:3], key=lambda t: score(t, STAGE2, [77]))
    assert chosen != ranked[0]  # stage 2, not stage 1, decides
    assert winner["inference_parameters"]["algorithm.predictor.temperature"] == chosen
    assert winner["score_interval"]["low"] < winner["score"] < winner["score_interval"]["high"]
    selected = json.loads(next((tmp_path / "runs/example/training/candidates").glob(
        "*/inference/native/selected.json")).read_text())
    assert [len(selected["two_stage"][k]) for k in ("stage1", "stage2")] == [12, 3]
    tests = [i for i, c in enumerate(recorder.calls) if c["evaluation"]["indices"] == TEST]
    assert tests == [len(recorder.calls) - 1]  # the test split once, last, for the winner
    assert recorder.calls[-1]["evaluation"]["seeds"] == [1041]
    assert "two_stage" not in recorder.calls[-1]["evaluation"]
    assert len(recorder.calls) == 12 + 3 + 1
    count = len(recorder.calls)
    assert tuning.run_case(spec(tmp_path), tmp_path / "runs", concurrency=2) == winner
    assert len(recorder.calls) == count  # an idempotent resume


@pytest.mark.parametrize("fail_at", [5, 13, 15])  # stage 1, stage 2, the test call
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
    resumed = tuning.run_case(settings, tmp_path / "b/runs", concurrency=1)
    assert resumed["inference_parameters"] == expected["inference_parameters"]
    assert resumed["score"] == expected["score"]
    assert len(interrupted.calls) == len(reference.calls)  # nothing finished is redone


def test_two_stage_is_opt_in_validated_and_costed(tmp_path):
    plain = base_spec(tmp_path, fixed=True)
    tuning.validate(plain)
    assert tuning.generation_cost(plain)["tuning_sequences"] == 2 * 2 * 1
    cost = tuning.generation_cost(spec(tmp_path))
    assert cost["tuning_sequences"] == 12 * 8 * 1 + 3 * 16 * 1
    for broken in (
        {"stage1": 0},
        {"stage2": 40},  # 8 + 40 exceeds the 40-prompt pool
        {"stage2_seeds": [41]},  # not fresh
        {"stage2_seeds": [1041]},
        {"stage2_overrides": {"sampler.steps": 3}},
    ):
        bad = spec(tmp_path)
        bad["evaluation"]["two_stage"].update(broken)
        with pytest.raises(ValueError):
            tuning.validate(bad)


@pytest.mark.parametrize("search,sampler", [
    ({"algorithm": "gp"}, "GPSampler"),
    ({"algorithm": "tpe", "multivariate": True, "group": True}, "TPESampler"),
])
def test_per_policy_optuna_sampler_is_honoured_and_resumes(tmp_path, monkeypatch, search, sampler):
    import optuna

    created = []
    original = tuning.make_sampler

    def recording(settings, completed_trials=0):
        made = original(settings, completed_trials)
        created.append((settings.get("search"), type(made).__name__, completed_trials))
        return made

    monkeypatch.setattr(tuning, "make_sampler", recording)
    settings = spec(tmp_path)
    policy = settings["inference"][0]
    policy.update(space={"algorithm.predictor.temperature": {"type": "float", "low": 0.5,
                                                              "high": 1.5}},
                  trials=14, initial_points=[{"algorithm.predictor.temperature": 1.0}],
                  search=search, on_failure="stop")
    interrupted = Recorder(fail_at=12)  # inside stage 1, after GP's 10 startup trials
    monkeypatch.setattr(tuning, "execute", interrupted)
    with pytest.raises(Exception, match="worker lost"):
        tuning.run_case(settings, tmp_path / "runs", concurrency=1)
    winner = tuning.run_case(settings, tmp_path / "runs", concurrency=1)
    assert {name for s, name, _ in created if s == search} == {sampler}
    assert any(count > 0 for s, _, count in created if s == search)  # reseeded on resume
    study = optuna.load_study(storage=f"sqlite:///{next((tmp_path / 'runs').rglob('native/optuna.sqlite'))}",
                              study_name="search")
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(complete) == 14
    calls = len(interrupted.calls)
    assert tuning.run_case(settings, tmp_path / "runs", concurrency=1) == winner
    assert len(interrupted.calls) == calls


def test_extra_test_budget_scores_every_policy_winner_after_selection(tmp_path, monkeypatch):
    settings = spec(tmp_path)
    second = copy.deepcopy(settings["inference"][0])
    second["name"] = "other"
    settings["inference"].append(second)
    settings["evaluation"].update(extra_test_nfe=[16], extra_test_examples=4)
    recorder = Recorder()
    monkeypatch.setattr(tuning, "execute", recorder)
    winner = tuning.run_case(settings, tmp_path / "runs", concurrency=1)
    tests = [(i, c) for i, c in enumerate(recorder.calls)
             if set(c["evaluation"]["indices"]) <= set(TEST)]
    # One full test call at the tuning budget for the winner, then one smoke call per policy.
    assert [c["evaluation"]["nfe"] for _, c in tests] == [4, 16, 16]
    n = len(recorder.calls)
    assert [i for i, _ in tests] == [n - 3, n - 2, n - 1]  # after every validation call
    assert tests[0][1]["evaluation"]["indices"] == TEST
    subsets = [c["evaluation"]["indices"] for _, c in tests[1:]]
    assert subsets[0] == subsets[1] and len(subsets[0]) == 4  # one fixed subset for all
    assert all(c["evaluation"]["seeds"] == [1041] for _, c in tests[1:])  # one test seed
    assert {c["policy"]["name"] for _, c in tests[1:]} == {"native", "other"}
    for _, call in tests[1:]:
        chosen = winner["policy_winners"][call["policy"]["name"]]["inference_parameters"]
        assert call["parameters"] == chosen  # each policy's own validation winner
    root = tmp_path / "runs/example"
    extra = json.loads((root / "final-extra/nfe-16/other/result.json").read_text())
    assert extra["measurement"] == "extra_test_nfe" and extra["nfe"] == 16
    # The extra numbers never reach selection: the case's choice is unchanged.
    plain = spec(tmp_path / "plain")
    plain["inference"].append(copy.deepcopy(second))
    monkeypatch.setattr(tuning, "execute", Recorder())
    reference = tuning.run_case(plain, tmp_path / "plain/runs", concurrency=1)
    assert reference["inference_parameters"] == winner["inference_parameters"]
    count = len(recorder.calls)
    monkeypatch.setattr(tuning, "execute", recorder)
    tuning.run_case(settings, tmp_path / "runs", concurrency=1)
    assert len(recorder.calls) == count
    cost = tuning.generation_cost(settings)
    assert cost["extra_test_calls"] == 2 * 4 * 1 * 16
    assert tuning.extra_test_subset(settings["evaluation"])[0] == subsets[0]  # deterministic
    assert cost["network_calls"] == cost["tuning_calls"] + cost["test_calls"] + cost["extra_test_calls"]
    for bad, message in [({"extra_test_nfe": [4]}, "extra_test_nfe"),
                         ({"extra_test_examples": 0}, "extra_test_examples"),
                         ({"extra_test_seeds": [41]}, "extra_test_seeds")]:
        broken = copy.deepcopy(settings)
        broken["evaluation"].update(bad)
        with pytest.raises(ValueError, match=message):
            tuning.validate(broken)


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
    assert len(recorder.calls) - 2 == 2 * (12 + 3)
    assert math.isclose(tuning.generation_cost(spec(tmp_path, nfe=[4, 8]))["network_calls"],
                        (12 * 8 + 3 * 16 + 10) * (4 + 8))


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


def test_test_score_and_overrides_reach_only_the_test_split(tmp_path, monkeypatch):
    settings = spec(tmp_path)
    settings["evaluation"].update(
        score={"kind": "metric", "metric": "ref_w1"},
        test_score={"kind": "metric", "metric": "sol"},
        overrides={"task.ref_w1": {"fit_indices": [1]}},
        test_overrides={"task.sol": {"n_directions": 8}, "task.primary_metric": "sol"},
        extra_test_nfe=[16], extra_test_examples=4,
    )
    recorder = Recorder()
    monkeypatch.setattr(tuning, "execute", recorder)
    tuning.run_case(settings, tmp_path / "runs", concurrency=1)
    tests = [c for c in recorder.calls if set(c["evaluation"]["indices"]) <= set(TEST)]
    tuned = [c for c in recorder.calls if c not in tests]
    assert len(tests) == 2 and tuned
    for call in tuned:  # selection reads the tuning metric, never the test settings
        assert call["evaluation"]["score"]["metric"] == "ref_w1"
        assert "task.sol" not in call["evaluation"].get("overrides", {})
        assert "test_score" not in call["evaluation"] and "test_overrides" not in call["evaluation"]
    for call in tests:
        assert call["evaluation"]["score"]["metric"] == "sol"
        assert call["evaluation"]["overrides"] == {
            "task.ref_w1": {"fit_indices": [1]}, "task.sol": {"n_directions": 8},
            "task.primary_metric": "sol"}


def test_per_example_bootstrap_of_stage_two_scores(tmp_path):
    records = [{"index": i, "exact": bool(i % 3)} for i in range(30)]
    score = {"kind": "metric", "metric": "accuracy", "per_example": "exact"}
    values = tuning.per_example_scores({"records": records}, score, 20 / 30)
    assert sum(values.values()) == 20
    with pytest.raises(ValueError, match="averages"):
        tuning.per_example_scores({"records": records}, score, 0.5)
    interval = tuning.mean_bootstrap([values], 200)
    assert interval["low"] < 20 / 30 < interval["high"] and interval["replicates"] == 200
    assert tuning.mean_bootstrap([values], 200) == interval  # seeded
    same = tuning.mean_bootstrap([{i: 1.0 for i in range(5)}], 50)
    assert same["low"] == same["high"] == 1.0
    settings = spec(tmp_path)
    settings["evaluation"]["two_stage"]["stage2_bootstrap"] = 200
    with pytest.raises(ValueError, match="per_example"):
        tuning.validate(settings)
    settings["evaluation"]["score"]["per_example"] = "exact"
    tuning.validate(settings)

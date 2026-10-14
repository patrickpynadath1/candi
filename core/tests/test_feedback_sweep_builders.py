"""Small feedback sweeps compose only the families they actually run."""

from examples import continuous_hybrid_feedback_sweep as continuous
from examples import discrete_feedback_sweep as discrete
from examples import masked_detach_depth_sweep as detached


def test_detached_sweep_composes_one_masked_recipe(monkeypatch):
    calls = []
    compose = continuous.compose

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return compose(*args, **kwargs)

    monkeypatch.setattr(continuous, "compose", record)
    jobs = detached.build_jobs("test")
    assert [args[0] for args, _ in calls if args] == ["sudoku_hard"]
    assert len(calls) == 2  # The recipe and its ancestral sampler.
    assert [job.name for job in jobs] == [
        "masked-token-detached-depth3",
        "masked-token-detached-depth0",
    ]
    for depth, job in zip((3, 0), jobs):
        assert job.config["trainer"]["max_steps"] == 5000
        assert job.config["token_feedback"]["depth"] == depth
        assert job.config["token_feedback"]["detach"] is True
        assert len(job.inference) == 4


def test_discrete_sweep_does_not_compose_continuous_families(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError(
            "Discrete sweep constructed an unrelated continuous recipe"
        )

    monkeypatch.setattr(continuous, "continuous_config", unexpected)
    jobs = discrete.build_jobs("test")
    assert len(jobs) == 9
    assert sum(len(job.inference) for job in jobs) == 48
    assert all(job.config["trainer"]["max_steps"] == 5000 for job in jobs)

"""Spherical flow matching on the view path matches the frozen standalone class.

The reference pair is core/tests/reference_algorithms/spherical_flow_matching.py
with its own frozen sampler (core/tests/reference_samplers/spherical.py). Every
check below holds with torch.equal: the view path computes the same tensors in
the same order and draws random numbers in the same order.
"""

from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest
import torch

from core.models.legacy import convert_state_dict
from core.samplers.spherical import SphericalFlowSampler
from core.tests.reference_samplers.spherical import (
    SphericalFlowSampler as ReferenceSampler,
)
from core.tests.view_equivalence import assert_training_identical, batch, build

REFERENCE = "core.tests.reference_algorithms.spherical_flow_matching.SphericalFlowMatching"
PRESETS = [
    "spherical_flow_matching",
    "sudoku_spherical",
    "sudoku_spherical_adaptive",
    "sudoku_spherical_truncated_adaptive",
]
SMALL = {"representation.dimension": 32}
# Refit the adaptive spline after every observation, so the second step and
# sampling run on a fitted (non-base) schedule.
FAST_ADAPTATION = {
    "noise_schedule.buffer_size": 8,
    "noise_schedule.warmup_steps": 0,
    "noise_schedule.refit_every": 1,
    "noise_schedule.n_knots": 4,
}
VARIANTS = {
    "default": {},
    "temperature": {"model.learn_temperature_scaling": True},
    "float32_process": {"process.precision": "float32"},
}
SAMPLERS = [
    {"steps": 6},  # preset settings: exact velocity, float64
    {"steps": 6, "use_float64": False},
    {"steps": 6, "velocity": "sample"},
    {"steps": 1},  # only the terminal prediction
]


def overrides(experiment, variant):
    options = {**SMALL, **VARIANTS[variant]}
    if "adaptive" in experiment:
        options.update(FAST_ADAPTATION)
    return options


def pair(experiment, variant, self_conditioning, seed=0):
    options = overrides(experiment, variant)
    cfg, reference = build(experiment, REFERENCE, self_conditioning, overrides=options)
    _, current = build(experiment, None, self_conditioning, seed=seed, overrides=options)
    return cfg, reference, current


def converted(state_dict):
    """The standalone state dict under the view-path names."""
    renamed = convert_state_dict(state_dict, "core.models.spherical.SphereDiT", "model.")
    view = ("schedule.", "representation.")
    return {
        (f"own_views.0.{key}" if key.startswith(view) else key): value
        for key, value in renamed.items()
    }


def sample(algorithm, sampler, prompt, fixed, seed=11):
    torch.manual_seed(seed)
    return algorithm.generate(sampler, prompt, fixed)


def assert_sampling_identical(reference, current, cfg):
    tokens, _ = batch(cfg)
    prompt = tokens[:3]
    fixed = torch.zeros_like(prompt, dtype=torch.bool)
    fixed[:, :20] = True
    reference.eval()
    current.eval()
    for options in SAMPLERS:
        expected, expected_info = sample(reference, ReferenceSampler(**options), prompt, fixed)
        actual, actual_info = sample(current, SphericalFlowSampler(**options), prompt, fixed)
        assert torch.equal(expected, actual), options
        assert expected_info == actual_info, options
        assert actual_info["nfe"] == options["steps"]


@pytest.mark.parametrize("self_conditioning", [False, True])
@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("experiment", PRESETS)
def test_training_and_adaptive_feedback_are_identical(experiment, variant, self_conditioning):
    cfg, reference, current = pair(experiment, variant, self_conditioning)
    tokens, mask = batch(cfg)
    for step in range(2):  # the second step sees the refit adaptive schedule
        assert_training_identical(reference, current, tokens, mask, seed=7 + step, step=step)
        for algorithm in (reference, current):
            algorithm.zero_grad()
    if "adaptive" in experiment:  # identical observations and refits
        old, new = reference.schedule.state_dict(), current.schedule.state_dict()
        assert old.keys() == new.keys()
        assert all(torch.equal(old[key], new[key]) for key in old)
        assert current.schedule.refit_count >= 1


@pytest.mark.parametrize("self_conditioning", [False, True])
@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("experiment", PRESETS)
def test_sampling_is_identical(experiment, variant, self_conditioning):
    cfg, reference, current = pair(experiment, variant, self_conditioning)
    if "adaptive" in experiment:  # sample on a fitted schedule
        tokens, mask = batch(cfg)
        assert_training_identical(reference, current, tokens, mask)
    assert_sampling_identical(reference, current, cfg)


@pytest.mark.parametrize("self_conditioning", [False, True])
@pytest.mark.parametrize("experiment", ["spherical_flow_matching", "sudoku_spherical_truncated_adaptive"])
def test_old_state_dicts_load_strictly(experiment, self_conditioning):
    cfg, reference, _ = pair(experiment, "default", self_conditioning)
    if "adaptive" in experiment:
        tokens, mask = batch(cfg)
        reference.train()
        reference.loss(tokens, mask, training=True)
    _, current = build(experiment, None, self_conditioning, seed=123, overrides=overrides(experiment, "default"))
    state = converted(reference.state_dict())
    assert state.keys() == current.state_dict().keys()
    current.load_state_dict(state, strict=True)
    assert all(
        torch.equal(value, current.state_dict()[key]) for key, value in state.items()
    )
    assert_sampling_identical(reference, current, cfg)


def test_constructor_checks_are_kept():
    from core.algorithms.spherical_flow_matching import SphericalFlowMatching

    cfg, _ = build("spherical_flow_matching", None, overrides=SMALL)
    parts = {
        name: instantiate(cfg.algorithm[name])
        for name in ("model", "predictor", "process", "schedule", "time_sampler", "objective", "representation")
    }
    with pytest.raises(Exception, match="dimensions must match"):
        instantiate(cfg.algorithm, model={"state_dim": 16})
    with pytest.raises(ValueError, match="keep coordinates"):
        SphericalFlowMatching(**{**parts, "schedule": object()})
    unnormalized = OmegaConf.merge(cfg.algorithm.representation, {"unit_normalized": False})
    with pytest.raises(ValueError, match="unit-normalized"):
        SphericalFlowMatching(**{**parts, "representation": instantiate(unnormalized)})


def test_view_path_components():
    _, algorithm = build("sudoku_spherical_truncated_adaptive", None, True, overrides=SMALL)
    assert algorithm.model.time.coordinate == "negative_log_keep_floor"
    assert algorithm.model.self_conditioning.space == "state"
    assert algorithm.views[0].schedule is algorithm.schedule
    assert algorithm.schedule.base.base.alpha_max < 1  # truncation is kept

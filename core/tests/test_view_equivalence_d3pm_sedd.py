"""D3PM and SEDD absorbing on the view path match their frozen standalone classes.

Each case builds the frozen class (core/tests/reference_algorithms) and the
current one from the same preset and seed. Parameters, loss (including D3PM's
clean-input reconstruction term), gradients, sampled tokens and NFE are
compared exactly: no random draw or float operation was reordered.
"""

from hydra.utils import instantiate
import pytest
import torch

from core.models.legacy import convert_state_dict
from core.tests.view_equivalence import (
    assert_sampling_identical,
    assert_training_identical,
    batch,
    build,
)

REFERENCE = {
    "d3pm_absorb": "core.tests.reference_algorithms.d3pm_absorb.D3PMAbsorb",
    "sedd_absorb": "core.tests.reference_algorithms.sedd_absorb.SeddAbsorb",
}
ANCESTRAL = "core.samplers.ancestral.AncestralSampler"
ANALYTIC = "core.samplers.analytic.AnalyticSampler"
OLD_ANALYTIC = "core.tests.reference_samplers.analytic_sedd.AnalyticSampler"
SAMPLERS = {
    # The preset: ancestral steps and an ancestral step to the clean endpoint.
    "d3pm_absorb": [
        {"_target_": ANCESTRAL, "steps": 8, "epsilon": 1e-3, "final_denoise": "ancestral"},
        # Settings the sampler group exposes; the bound never clamps observed
        # tokens, so these check that generation still keeps them.
        {"_target_": ANCESTRAL, "steps": 8, "epsilon": 1e-3, "final_denoise": "greedy"},
        {
            "_target_": ANCESTRAL,
            "steps": 8,
            "epsilon": 1e-3,
            "final_denoise": "greedy",
            "greedy_clean_estimate": True,
        },
    ],
    # The preset (float64 transitions, ancestral terminal step) and the other
    # options the sampler exposes.
    "sedd_absorb": [
        {"steps": 8, "epsilon": 1e-3, "final_denoise": "ancestral", "use_float64": True},
        {"steps": 8, "epsilon": 1e-3, "final_denoise": "ancestral", "use_float64": False},
        {"steps": 8, "epsilon": 1e-3, "final_denoise": "none", "use_float64": True},
    ],
}


def pair(experiment, seed=0):
    cfg, reference = build(experiment, REFERENCE[experiment])
    _, current = build(experiment, None, seed=seed)
    return cfg, reference, current


def converted_state_dict(reference):
    state = convert_state_dict(reference.state_dict(), "core.models.dit.DiT", "model.")
    # The token table belongs to the view's representation on the view path.
    state["model.input.views.0.representation.embedding"] = state.pop(
        "model.input.embedding.embedding"
    )
    return state


@pytest.mark.parametrize("experiment", sorted(REFERENCE))
def test_training_is_identical(experiment):
    cfg, reference, current = pair(experiment)
    assert_training_identical(reference, current, *batch(cfg))


def test_d3pm_reconstruction_term_is_identical():
    # Every token observed: only the clean-input reconstruction pass remains.
    cfg, reference, current = pair("d3pm_absorb")
    tokens, mask = batch(cfg)
    results = assert_training_identical(
        reference, current, tokens, mask, corruption_mask=torch.zeros_like(mask)
    )
    assert results[0]["loss"] > 0


@pytest.mark.parametrize("experiment", sorted(REFERENCE))
def test_converted_checkpoint_loads_and_samples_identically(experiment):
    cfg, reference, _ = pair(experiment)
    _, current = build(experiment, None, seed=123)
    current.load_state_dict(converted_state_dict(reference), strict=True)
    tokens, _ = batch(cfg)
    prompt = tokens[:3]
    fixed = torch.zeros_like(prompt, dtype=torch.bool)
    fixed[:, :6] = True
    if experiment == "d3pm_absorb":
        assert_sampling_identical(reference, current, prompt, fixed, SAMPLERS[experiment])
        return
    # The analytic sampler's call form changed: the frozen sampler drives the
    # frozen class.
    reference.eval()
    current.eval()
    for options in SAMPLERS[experiment]:
        torch.manual_seed(11)
        expected, expected_info = reference.generate(
            instantiate({"_target_": OLD_ANALYTIC, **options}), prompt, fixed
        )
        torch.manual_seed(11)
        actual, actual_info = current.generate(
            instantiate({"_target_": ANALYTIC, **options}), prompt, fixed
        )
        assert torch.equal(expected, actual), options
        assert expected_info == actual_info, options


def test_sedd_adaptive_objective_feedback_is_identical():
    # Score adaptation is allowed with loss_signal=objective only.
    overrides = {
        "noise_schedule": "cdcd_adaptive",
        "noise_schedule.loss_signal": "objective",
        "noise_schedule.warmup_steps": 0,
    }
    cfg, reference = build("sedd_absorb", REFERENCE["sedd_absorb"], overrides=overrides)
    _, current = build("sedd_absorb", None, overrides=overrides)
    assert_training_identical(reference, current, *batch(cfg), step=3)
    old, new = reference.schedule.state_dict(), current.schedule.state_dict()
    assert old.keys() == new.keys()
    assert all(torch.equal(old[key], new[key]) for key in old)
    assert old["num_observed"] > 0

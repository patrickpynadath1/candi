"""D3PM and SEDD absorbing on the view path match their frozen standalone classes.

Each case builds the frozen class (core/tests/reference_algorithms) and the
current one from the same preset and seed. Parameters, loss (including D3PM's
clean-input reconstruction term), gradients, sampled tokens and NFE are
compared exactly: no random draw or float operation was reordered.
"""

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
}
ANCESTRAL = "core.samplers.ancestral.AncestralSampler"
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
    assert_sampling_identical(reference, current, prompt, fixed, SAMPLERS[experiment])

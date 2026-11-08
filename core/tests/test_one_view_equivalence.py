"""Standalone families moved onto the hybrid path must be bit-identical.

Each case builds the frozen standalone implementation (core/tests/
reference_algorithms) and the current class from the same config and seed, then
requires identical initialization, parameter order (EMA and optimizer state are
stored by position), loss, gradients, old-checkpoint loading and samples.
"""

from hydra.utils import instantiate
import pytest
import torch

from core.tests.view_equivalence import (
    assert_sampling_identical,
    assert_training_identical,
    batch,
    build,
)

FAMILIES = {
    "uniform_posterior": (
        "core.tests.reference_algorithms.uniform_posterior.PosteriorUniformDiffusion",
        [
            ("core.samplers.uniform_ancestral.UniformAncestralSampler", {"final_denoise": "ancestral"}),
            ("core.samplers.uniform_ancestral.UniformAncestralSampler", {"final_denoise": "greedy"}),
            (
                "core.samplers.uniform_ancestral.UniformAncestralSampler",
                {"final_denoise": "ancestral", "use_float64": False},
            ),
        ],
    ),
}


def pair(experiment, self_conditioning):
    reference_target, _ = FAMILIES[experiment]
    cfg, reference = build(experiment, reference_target, self_conditioning)
    _, current = build(experiment, None, self_conditioning)
    return cfg, reference, current


@pytest.mark.parametrize("self_conditioning", [False, True])
@pytest.mark.parametrize("experiment", sorted(FAMILIES))
def test_initialization_and_parameter_order_are_identical(experiment, self_conditioning):
    _, reference, current = pair(experiment, self_conditioning)
    old, new = list(reference.parameters()), list(current.parameters())
    assert len(old) == len(new)
    assert all(a.shape == b.shape and torch.equal(a, b) for a, b in zip(old, new))


@pytest.mark.parametrize("self_conditioning", [False, True])
@pytest.mark.parametrize("experiment", sorted(FAMILIES))
def test_loss_and_gradients_are_identical(experiment, self_conditioning):
    cfg, reference, current = pair(experiment, self_conditioning)
    tokens, mask = batch(cfg)
    results = []
    for algorithm in (reference, current):
        algorithm.train()
        torch.manual_seed(7)
        result = algorithm.loss(tokens, mask, training=True)
        result["loss"].backward()
        results.append(result)
    assert torch.equal(results[0]["loss"], results[1]["loss"])
    for a, b in zip(reference.parameters(), current.parameters()):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            assert torch.equal(a.grad, b.grad)


@pytest.mark.parametrize("self_conditioning", [False, True])
@pytest.mark.parametrize("experiment", sorted(FAMILIES))
def test_old_checkpoints_load_and_sample_identically(experiment, self_conditioning):
    cfg, reference, _ = pair(experiment, self_conditioning)
    # A differently seeded current model must take every tensor from the old
    # standalone state dict, converted (core/models/legacy_checkpoints.py) and
    # loaded strictly, as when evaluating an existing checkpoint.
    from core.models.legacy_checkpoints import convert_checkpoint

    current_cfg, current = build(experiment, None, self_conditioning, seed=123)
    old = {"state_dict": {"algorithm." + k: v for k, v in reference.state_dict().items()}}
    converted = convert_checkpoint(old, current_cfg, algorithm=current)["state_dict"]
    current.load_state_dict({k.removeprefix("algorithm."): v for k, v in converted.items()}, strict=True)
    for a, b in zip(reference.parameters(), current.parameters()):
        assert torch.equal(a, b)
    tokens, _ = batch(cfg)
    prompt = tokens[:3]
    fixed = torch.zeros_like(prompt, dtype=torch.bool)
    fixed[:, :6] = True
    reference.eval()
    current.eval()
    for target, options in FAMILIES[experiment][1]:
        sampler = instantiate({"_target_": target, "steps": 8, "epsilon": 1e-3, **options})
        torch.manual_seed(11)
        expected, expected_info = reference.generate(sampler, prompt, fixed)
        torch.manual_seed(11)
        actual, actual_info = current.generate(sampler, prompt, fixed)
        assert torch.equal(expected, actual), options
        assert expected_info["nfe_per_sample"] == actual_info["nfe_per_sample"]


MDLM_VARIANTS = {
    "default": {},
    "adaln_zero_time": {"model.adaLN": True},
    "adaln_negative_log_keep": {"model.adaLN": True, "algorithm.time_conditioning": True},
    "cross_entropy": {"objective": "cross_entropy"},
    "self_conditioning": {
        "model.adaLN": True,
        "self_conditioning.enabled": True,
        "self_conditioning.source": "posterior_mean",
    },
}


MDLM_REFERENCE = "core.tests.reference_algorithms.mdlm.MDLM"


@pytest.mark.parametrize("variant", sorted(MDLM_VARIANTS))
def test_mdlm_training_is_identical_on_the_view_path(variant):
    cfg, standalone = build("sudoku_hard", MDLM_REFERENCE, overrides=MDLM_VARIANTS[variant])
    _, viewed = build("sudoku_hard", None, overrides=MDLM_VARIANTS[variant])
    assert_training_identical(standalone, viewed, *batch(cfg))


MDLM_SAMPLERS = [
    {"_target_": "core.samplers.ancestral.AncestralSampler", "steps": 8, "final_denoise": "greedy"},
    {"_target_": "core.samplers.ancestral.AncestralSampler", "steps": 8, "final_denoise": "ancestral"},
    {
        "_target_": "core.samplers.ancestral.AncestralSampler",
        "steps": 8,
        "final_denoise": "greedy",
        "greedy_clean_estimate": True,
    },
    {"_target_": "core.samplers.entropy_bounded.EntropyBoundedSampler", "entropy_bound": 0.3, "max_steps": 180},
]


@pytest.mark.parametrize("variant", ["default", "adaln_negative_log_keep", "self_conditioning"])
def test_mdlm_sampling_is_identical_on_the_view_path(variant):
    cfg, standalone = build("sudoku_hard", MDLM_REFERENCE, overrides=MDLM_VARIANTS[variant])
    _, viewed = build("sudoku_hard", None, overrides=MDLM_VARIANTS[variant])
    tokens, _ = batch(cfg)
    prompt = tokens[:3]
    fixed = torch.zeros_like(prompt, dtype=torch.bool)
    fixed[:, :20] = True
    assert_sampling_identical(standalone, viewed, prompt, fixed, MDLM_SAMPLERS)

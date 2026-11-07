"""BlurringDiffusion on the view path behaves as the frozen standalone class.

The reference (core/tests/reference_algorithms/blurring_diffusion.py) and the
current class are built from the same preset config and seed. ContinuousDiT
held bare root parameters (input/output embeddings), so the adopted ModularDiT
lists parameters in another order: parameters are paired by value
(``ordered=False``) and checkpoints by converted name. Training, sampling
(including bootstrap/previous self-conditioning and greedy clean estimates),
NFE and time-sampler observations all match exactly.
"""

import pytest
import torch
from omegaconf import OmegaConf

from core.models.legacy import convert_state_dict
from core.modifiers.block_diffusion import BlockDiffusion
from core.samplers.blur_ancestral import BlurAncestralSampler
from core.tests.reference_samplers.blur_ancestral import (
    BlurAncestralSampler as ReferenceSampler,
)
from core.tests.view_equivalence import assert_training_identical, batch, build

REFERENCE = "core.tests.reference_algorithms.blurring_diffusion.BlurringDiffusion"
CONTINUOUS = "core.models.continuous_dit.ContinuousDiT"

# (experiment, self_conditioning, source, extra overrides); every option the
# three blurring presets use, plus detach_forward_embedding.
VARIANTS = {
    "sudoku": ("blurring_diffusion", False, None, {}),
    "sudoku_sc_posterior_mean": ("blurring_diffusion", True, "posterior_mean", {}),
    "sudoku_sc_hidden": ("blurring_diffusion", True, "hidden", {}),
    "sudoku_hard": ("sudoku_blurring_hard", False, None, {}),
    "sudoku_detached_forward": (
        "blurring_diffusion", False, None, {"algorithm.detach_forward_embedding": True}
    ),
    # Hidden SC at probability 0.9, adaptive CDCD time sampler, 128-dim codebook.
    "tinygsm": (
        "tinygsm_blurring_sc_adaptive",
        None,
        None,
        {"model.output_dim": 48, "task.length": 64},
    ),
    "tinygsm_sc_posterior_mean": (
        "tinygsm_blurring_sc_adaptive",
        None,
        "posterior_mean",
        {"model.output_dim": 48, "task.length": 64},
    ),
}

SAMPLERS = [
    {"steps": 3},
    {"steps": 3, "greedy_clean_estimate": True},
    {"steps": 3, "self_conditioning_mode": "bootstrap"},
    {"steps": 2, "greedy_clean_estimate": True, "self_conditioning_mode": "bootstrap"},
]


def pair(variant):
    experiment, sc, source, overrides = VARIANTS[variant]
    overrides = dict(overrides)
    if source is not None:
        overrides["self_conditioning.source"] = source
    cfg, reference = build(experiment, REFERENCE, sc, overrides=overrides)
    _, current = build(experiment, None, sc, overrides=overrides)
    return cfg, reference, current


def converted(reference):
    state = convert_state_dict(reference.state_dict(), CONTINUOUS, "model.")
    return {
        ("own_views.0." + key if key.startswith("representation.") else key): value
        for key, value in state.items()
    }


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_initialization_and_checkpoint_names(variant):
    _, reference, current = pair(variant)
    assert type(current.model).__name__ == "ModularDiT"
    state = converted(reference)
    assert sorted(state) == sorted(current.state_dict())
    for key, value in current.state_dict().items():
        assert torch.equal(state[key], value), key
    current.load_state_dict(state, strict=True)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_training_is_identical(variant):
    cfg, reference, current = pair(variant)
    tokens, mask = batch(cfg)
    assert_training_identical(reference, current, tokens, mask, ordered=False, step=3)
    # Adaptive time samplers observed the same per-token losses.
    if not hasattr(current.time_sampler, "state_dict"):
        return
    expected = reference.time_sampler.state_dict()
    actual = current.time_sampler.state_dict()
    assert expected.keys() == actual.keys() and actual["count"].item() > 0
    assert all(torch.equal(expected[k], actual[k]) for k in expected)
    # The codebook receives gradients through the clean state either way.
    assert current.representation.table.grad is not None


def test_detached_forward_embedding_stops_only_the_process_input():
    cfg, reference, current = pair("sudoku_detached_forward")
    tokens, mask = batch(cfg)
    assert_training_identical(reference, current, tokens, mask, ordered=False)
    assert current.detach_forward_embedding


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_sampling_is_identical(variant):
    cfg, reference, current = pair(variant)
    tokens, mask = batch(cfg)
    fixed = ~mask
    reference.eval()
    current.eval()
    for options in SAMPLERS:
        torch.manual_seed(11)
        expected, expected_info = reference.generate(ReferenceSampler(**options), tokens, fixed)
        torch.manual_seed(11)
        actual, actual_info = current.generate(BlurAncestralSampler(**options), tokens, fixed)
        assert torch.equal(expected, actual), options
        assert expected_info == actual_info, options
        assert torch.equal(actual[fixed], tokens[fixed])


@pytest.mark.parametrize("variant", ["sudoku", "sudoku_sc_posterior_mean", "sudoku_sc_hidden"])
def test_block_diffusion_is_identical(variant):
    cfg, reference, current = pair(variant)
    tokens, mask = batch(cfg, size=2)
    wrapped = [
        BlockDiffusion(algorithm, block_size=27, clean_loss="autoregressive", cache_clean_prefix=False)
        for algorithm in (reference, current)
    ]
    for algorithm in wrapped:
        algorithm.train()
    assert_training_identical(*wrapped, tokens, mask, ordered=False)
    fixed = ~mask
    for algorithm in wrapped:
        algorithm.eval()
    torch.manual_seed(11)
    expected, expected_info = wrapped[0].generate(ReferenceSampler(steps=2), tokens, fixed)
    torch.manual_seed(11)
    actual, actual_info = wrapped[1].generate(BlurAncestralSampler(steps=2), tokens, fixed)
    assert torch.equal(expected, actual)
    assert expected_info["nfe"] == actual_info["nfe"]


def test_unsupported_loss_overrides_stay_unavailable():
    cfg, _, current = pair("sudoku")
    tokens, mask = batch(cfg)
    for option in ("time_sampler", "objective", "pure_noise"):
        with pytest.raises(TypeError):
            current.loss(tokens, mask, **{option: None})


def test_constructor_checks_are_kept():
    from hydra.errors import InstantiationException
    from hydra.utils import instantiate

    cfg, _, _ = pair("sudoku")
    OmegaConf.update(cfg, "algorithm._target_", "core.algorithms.blurring_diffusion.BlurringDiffusion")
    for key, value, message in [
        ("algorithm.model.input_preconditioning", "gaussian", "raw time"),
        ("algorithm.model.time_conditioning", "log_one_minus_time", "raw time"),
        ("algorithm.model.state_dim", 8, "state_dim"),
        ("algorithm.objective", {"_target_": "types.SimpleNamespace", "requires_context": True}, "cross entropy"),
    ]:
        bad = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        OmegaConf.update(bad, key, value)
        with pytest.raises(InstantiationException, match=message):
            instantiate(bad.algorithm)


def test_bf16_autocast_training_is_identical():
    """TinyGSM trains under bf16-mixed; the diffusion state stays float32."""
    cfg, reference, current = pair("tinygsm")
    tokens, mask = batch(cfg)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert_training_identical(reference, current, tokens, mask, ordered=False)

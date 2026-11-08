"""SimplexDiffusion on the view path is bit-identical to the standalone class.

Each case builds the frozen standalone class (core/tests/reference_algorithms/
simplex_diffusion.py) and the current one from the same preset and seed, and
requires torch.equal on initialization and parameter order, loss, every
gradient, adaptive time-sampler state, old-checkpoint loading, sampled tokens
and NFE. Sampling is compared twice: the frozen class with the frozen sampler
(core/tests/reference_samplers/simplex.py) against the current class with the
current sampler, and both classes under the current sampler.
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

REFERENCE = "core.tests.reference_algorithms.simplex_diffusion.SimplexDiffusion"
TINY_TASK = {"task.length": 16, "task.vocab_size": 97}

# (preset, overrides): every option the six simplex presets use, plus the
# posterior-mean feedback source and condition dropout the class supports.
VARIANTS = {
    "simplex": ("simplex", {}),
    "sudoku_simplex": ("sudoku_simplex", {}),
    "sudoku_simplex_adaptive": (
        "sudoku_simplex_adaptive",
        {"time_sampler.warmup": 2, "time_sampler.refit_every": 1},
    ),
    "sudoku_simplex_argmax": ("sudoku_simplex_argmax", {}),
    "sudoku_simplex_self_conditioning": ("sudoku_simplex_self_conditioning", {}),
    "tinygsm_simplex_sc": ("tinygsm_simplex_sc", TINY_TASK),
    "argmax_self_conditioning": (
        "sudoku_simplex_argmax",
        {"self_conditioning.enabled": True, "self_conditioning.probability": 0.9},
    ),
    "posterior_mean_self_conditioning": (
        "sudoku_simplex",
        {
            "self_conditioning.enabled": True,
            "self_conditioning.source": "posterior_mean",
            "self_conditioning.probability": 0.9,
        },
    ),
    "condition_dropout": (
        "sudoku_simplex_self_conditioning",
        {"algorithm.condition_dropout": 0.5},
    ),
    "no_adaLN": ("sudoku_simplex", {"model.adaLN": False}),
}

SAMPLER = "core.samplers.simplex.SimplexBridgeSampler"
REFERENCE_SAMPLER = "core.tests.reference_samplers.simplex.SimplexBridgeSampler"
SAMPLERS = [
    {"steps": 8},  # configs/sampler/simplex.yaml: churn 1, linear grid
    {"steps": 16},  # tinygsm_simplex_sc
    {"steps": 8, "churn": 0.0},
    {"steps": 8, "churn": 0.2},
    {"steps": 1},
    {"steps": 8, "confidence_power": 1.0},
]
GUIDED = [
    {"steps": 4, "guidance_scale": 2.0},
    {"steps": 4, "guidance_scale": 2.0, "confidence_power": 1.0},
    {"steps": 4, "guidance_scale": 0.0, "churn": 0.2},
]


def pair(variant, seed=0):
    experiment, overrides = VARIANTS[variant]
    cfg, reference = build(experiment, REFERENCE, overrides=overrides, seed=seed)
    _, current = build(experiment, None, overrides=overrides, seed=seed)
    return cfg, reference, current


def converted(reference):
    """The standalone state dict under the view recipe's names."""
    state = convert_state_dict(
        reference.state_dict(), "core.models.simplex.SimplexDiT", prefix="model."
    )
    return {
        ("own_views.0." + key if key.startswith("process.") else key): value
        for key, value in state.items()
    }


def samplers(variant):
    guided = VARIANTS[variant][1].get("algorithm.condition_dropout", 0) > 0
    return SAMPLERS + (GUIDED if guided else [])


def prompt_of(cfg):
    tokens, _ = batch(cfg)
    prompt = tokens[:3]
    fixed = torch.zeros_like(prompt, dtype=torch.bool)
    fixed[:, :6] = True
    return prompt, fixed


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_training_is_identical_on_the_view_path(variant):
    cfg, reference, current = pair(variant)
    tokens, mask = batch(cfg)
    assert_training_identical(reference, current, tokens, mask)
    for step in range(1, 4):  # Adaptive time samplers observe the same losses.
        for algorithm in (reference, current):
            torch.manual_seed(20 + step)
            algorithm.loss(tokens, mask, training=True, step=step)
    old, new = reference.time_sampler.state_dict(), current.time_sampler.state_dict()
    assert old.keys() == new.keys()
    assert all(torch.equal(old[key], new[key]) for key in old)
    # Validation-style losses (no condition dropout or feedback).
    for algorithm in (reference, current):
        algorithm.eval()
    results = []
    for algorithm in (reference, current):
        torch.manual_seed(5)
        with torch.no_grad():
            results.append(algorithm.loss(tokens, mask))
    for key in ("loss", "loss_sum", "num_tokens"):
        assert torch.equal(results[0][key], results[1][key])


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_sampling_is_identical_on_the_view_path(variant):
    cfg, reference, current = pair(variant)
    prompt, fixed = prompt_of(cfg)
    reference.eval()
    current.eval()
    for options in samplers(variant):
        old_sampler = instantiate({"_target_": REFERENCE_SAMPLER, **options})
        new_sampler = instantiate({"_target_": SAMPLER, **options})
        torch.manual_seed(11)
        expected, expected_info = reference.generate(old_sampler, prompt, fixed)
        torch.manual_seed(11)
        actual, actual_info = current.generate(new_sampler, prompt, fixed)
        assert torch.equal(expected, actual), options
        assert expected_info == actual_info, options
    assert_sampling_identical(
        reference,
        current,
        prompt,
        fixed,
        [{"_target_": SAMPLER, **options} for options in samplers(variant)],
    )


def test_adaptive_time_grid_sampling_is_identical():
    cfg, reference, current = pair("sudoku_simplex_adaptive")
    tokens, mask = batch(cfg)
    for step in range(4):
        for algorithm in (reference, current):
            algorithm.train()
            torch.manual_seed(30 + step)
            algorithm.loss(tokens, mask, training=True, step=step)
    assert reference.time_sampler.ready and current.time_sampler.ready
    prompt, fixed = prompt_of(cfg)
    assert_sampling_identical(
        reference,
        current,
        prompt,
        fixed,
        [{"_target_": SAMPLER, "steps": 8, "time_grid": "adaptive"}],
    )


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_old_checkpoints_load_strictly_and_sample_identically(variant):
    cfg, reference, _ = pair(variant)
    # A differently seeded current recipe takes every tensor from the
    # converted standalone state dict, as when evaluating an old checkpoint.
    _, current = build(VARIANTS[variant][0], None, overrides=VARIANTS[variant][1], seed=123)
    current.load_state_dict(converted(reference), strict=True)
    old, new = list(reference.parameters()), list(current.parameters())
    assert len(old) == len(new)
    assert all(torch.equal(a, b) for a, b in zip(old, new))
    prompt, fixed = prompt_of(cfg)
    assert_sampling_identical(
        reference, current, prompt, fixed, [{"_target_": SAMPLER, "steps": 8}]
    )


def test_checkpoint_keys_are_the_converted_standalone_keys():
    _, reference, current = pair("sudoku_simplex_self_conditioning")
    assert list(current.state_dict()) == list(converted(reference))


def test_view_recipe_keeps_the_standalone_attributes():
    cfg, _, current = pair("condition_dropout")
    assert current.condition_dropout == 0.5
    assert current.length == cfg.task.length
    assert current.process is current.views[0].process
    assert current.schedule is current.views[0].schedule
    assert current.model.input.table is current.prediction_embedding_table()
    with pytest.raises(Exception, match="condition_dropout must be in"):  # Hydra wraps it
        build("sudoku_simplex", None, overrides={"algorithm.condition_dropout": 1.5})

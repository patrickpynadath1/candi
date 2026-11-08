"""RePlaid on the view path matches the frozen standalone class.

Both presets (and their self-conditioning sources / prompt modes) are built
from the same config and seed as the frozen class
(core/tests/reference_algorithms/replaid.py) and the current one. Loss,
gradients, allocation and schedule feedback, samples, NFE and optimizer groups
are compared with torch.equal (exact on CPU). Parameters are paired by shape
and value (``ordered=False``) for gradients, and by converted name for the
optimizer groups: ContinuousDiT held bare root parameters and the view now
holds the representation and schedule.
"""

from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from core.tests.test_view_equivalence_gaussian import (
    assert_same_feedback,
    assert_sampling_pairs_identical,
    convert,
)
from core.tests.view_equivalence import assert_training_identical, batch, build
from core.training.module import TrainingModule

REFERENCE = "core.tests.reference_algorithms.replaid.RePlaid"
OLD_GROUPS = [
    {
        "patterns": ["representation.table", "schedule.input_weight", "schedule.bias", "schedule.output_weight"],
        "lr": 0.01,
        "weight_decay": 0.0,
    },
    {"patterns": ["schedule.gamma_min", "schedule.gamma_max"], "lr": 0.01, "weight_decay": 0.1},
]
SMALL = {"model.n_heads": 4}  # hidden 32 / 4 heads keeps an even head dimension.
CASES = {
    "replaid": ("replaid", {}),
    "replaid_sudoku_hard": ("replaid_sudoku_hard", {}),
    "replaid_hidden": ("replaid", {"self_conditioning.source": "hidden"}),
    "replaid_sc_all": ("replaid_sudoku_hard", {"self_conditioning.probability": 1.0}),
    "replaid_no_sc": ("replaid", {"self_conditioning.enabled": False}),
}


def pair(case, seed=0):
    experiment, overrides = CASES[case]
    overrides = {**SMALL, **overrides}
    cfg, reference = build(experiment, REFERENCE, overrides=overrides, seed=seed)
    _, current = build(experiment, None, overrides=overrides, seed=seed)
    return cfg, reference, current


@pytest.mark.parametrize("case", sorted(CASES))
def test_initialization_and_checkpoint_names(case):
    _, reference, current = pair(case)
    state = convert(reference.state_dict())
    assert state.keys() == current.state_dict().keys()
    for key, value in current.state_dict().items():
        assert torch.equal(value, state[key]), key
    experiment, overrides = CASES[case]
    _, other = build(experiment, None, overrides={**SMALL, **overrides}, seed=123)
    other.load_state_dict(state, strict=True)


@pytest.mark.parametrize("size", [1, 4])
@pytest.mark.parametrize("case", sorted(CASES))
def test_training_and_feedback_are_identical(case, size):
    cfg, reference, current = pair(case)
    tokens, mask = batch(cfg, size=size)
    for step in range(2):
        for algorithm in (reference, current):
            algorithm.zero_grad(set_to_none=True)
        results = assert_training_identical(
            reference, current, tokens, mask, seed=7 + step, ordered=False, step=step
        )
        for key in ("loss_sum", "num_tokens"):
            assert torch.equal(results[0][key], results[1][key])
        assert_same_feedback(reference, current)


def test_training_prediction_endpoint_is_identical():
    _, reference, current = pair("replaid")
    state = torch.randn(4, 5, 16)
    time = torch.linspace(0.1, 0.9, 4)
    path = reference.schedule(time)
    for algorithm in (reference, current):
        algorithm.train()
    torch.manual_seed(3)
    expected = reference.training_prediction(state, time, path)
    torch.manual_seed(3)
    actual = current.training_prediction((state,), current.points(time), time)
    assert torch.equal(expected, actual)


def _ddpm(module, **options):
    return {"_target_": f"{module}.GaussianDDPMSampler", "steps": 4, **options}


@pytest.mark.parametrize("case", sorted(CASES))
def test_sampling_is_identical(case):
    cfg, reference, current = pair(case)
    tokens, _ = batch(cfg)
    prompt = tokens[:3]
    fixed = torch.zeros_like(prompt, dtype=torch.bool)
    fixed[:, :6] = True
    preset = {k: v for k, v in OmegaConf.to_container(cfg.sampler).items() if k != "_target_"}
    settings = [
        {**preset, "steps": 4},
        {**preset, "steps": 4, "prompt_mode": "noisy"},
        {**preset, "steps": 4, "prompt_mode": "clean", "self_conditioning_mode": "previous"},
        {**preset, "steps": 4, "final_transition_noise": False, "prior_mode": "schedule"},
    ]
    pairs = [
        (
            _ddpm("core.tests.reference_samplers.gaussian_ddpm", **options),
            _ddpm("core.samplers.gaussian_ddpm", **options),
        )
        for options in settings
    ]
    assert_sampling_pairs_identical(reference, current, prompt, fixed, pairs)


@pytest.mark.parametrize("experiment", ["replaid", "replaid_sudoku_hard"])
def test_optimizer_groups_hold_the_same_tensors(experiment):
    cfg, reference, current = pair(experiment)
    old = TrainingModule.optimizer_parameters(
        SimpleNamespace(algorithm=reference), OmegaConf.create({"groups": OLD_GROUPS})
    )
    new = TrainingModule.optimizer_parameters(SimpleNamespace(algorithm=current), cfg.optimizer)
    assert len(old) == len(new) == 3
    # Pair tensors by their converted names (not by value: equal-valued
    # initializations, e.g. zero biases, would be ambiguous).
    current_named = dict(current.named_parameters())
    by_reference = {
        id(p): current_named[next(iter(convert({name: None})))]
        for name, p in reference.named_parameters()
    }
    for before, after in zip(old, new):
        assert {k: v for k, v in before.items() if k != "params"} == {
            k: v for k, v in after.items() if k != "params"
        }
        assert sorted(id(by_reference[id(p)]) for p in before["params"]) == sorted(
            id(p) for p in after["params"]
        )

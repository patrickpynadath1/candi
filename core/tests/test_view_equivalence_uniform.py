"""UniformDiffusion and Duo on the view path behave as their standalone forms.

Each case builds the frozen standalone class (core/tests/reference_algorithms/
uniform_diffusion.py, duo.py) and the current one from the same preset and
seed, with the preset's modifiers (block diffusion, block-size mixtures,
self-conditioning), then compares initialization, old-checkpoint loading, loss,
gradients, sampled tokens and NFE. Every comparison here is exact: the view
path computes the same operations in the same order and draws the same random
numbers.
"""

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from core.models.legacy import convert_state_dict
from core.noise_schedules.duo_calibration import build_integral_table
from core.sweeps import compose
from core.tests.view_equivalence import SMALL, assert_training_identical
from core.training.module import build_algorithm

REFERENCE = {
    "core.algorithms.uniform_diffusion.UniformDiffusion": (
        "core.tests.reference_algorithms.uniform_diffusion.UniformDiffusion"
    ),
    "core.algorithms.duo.Duo": "core.tests.reference_algorithms.duo.Duo",
}

# preset -> overrides; every option the uniform presets use.
VARIANTS = {
    "uniform_diffusion": ("uniform_diffusion", {}),
    "udlm": ("udlm", {}),
    "uniform_zero_time": ("uniform_diffusion", {"+algorithm.time_conditioning": False}),
    "uniform_sc": ("uniform_diffusion", {"self_conditioning.enabled": True}),
    "blockgen_uniform": ("blockgen_uniform", {"block_diffusion.block_size": 4}),
    "blockgen_uniform_mixture": (
        "blockgen_uniform_mixture",
        {"block_diffusion.block_size": 4, "block_diffusion.training.sizes": [1, 4]},
    ),
    "tinygsm_blockgen_uniform": (
        "tinygsm_blockgen_uniform",
        {"task.length": 16, "block_diffusion.block_size": 4},
    ),
    "tinygsm_uniform_sc": ("tinygsm_uniform_sc", {"task.length": 16}),
    "duo": ("duo", {}),
    "duo_sc": ("duo", {"self_conditioning.enabled": True}),
}


@pytest.fixture(scope="module")
def duo_table(tmp_path_factory):
    tables = {}

    def path(vocab_size):
        if vocab_size not in tables:
            file = tmp_path_factory.mktemp("duo") / f"v{vocab_size}.npz"
            np.savez(file, **build_integral_table(vocab_size, num_points=10001))
            tables[vocab_size] = str(file)
        return tables[vocab_size]

    return path


def config(variant, duo_table):
    experiment, overrides = VARIANTS[variant]
    options = {**SMALL, **overrides}
    if experiment == "duo":
        options["curriculum.start_step"] = 1
        options["curriculum.end_step"] = 3
    cfg = compose(experiment, [f"{key}={value}" for key, value in options.items()])
    if experiment == "duo":
        OmegaConf.update(cfg, "calibration.cache_path", duo_table(cfg.task.vocab_size))
    return cfg


def pair(variant, duo_table, seed=0):
    cfg = config(variant, duo_table)
    reference_cfg = cfg.copy()
    OmegaConf.update(
        reference_cfg, "algorithm._target_", REFERENCE[cfg.algorithm._target_]
    )
    torch.manual_seed(seed)
    reference = build_algorithm(reference_cfg)
    torch.manual_seed(seed)
    current = build_algorithm(cfg)
    return cfg, reference, current


def batch(cfg, size=4, prompt=5):
    torch.manual_seed(1)
    tokens = torch.randint(0, cfg.task.vocab_size, (size, cfg.task.length))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, :prompt] = False
    return tokens, mask


def converted(state_dict):
    """A standalone checkpoint renamed for the view network (the DiT sits
    under ``model.``, or ``algorithm.model.model.`` inside block diffusion)."""
    prefix = next(k for k in state_dict if k.endswith("vocab_embed.embedding"))[: -len("vocab_embed.embedding")]
    state = convert_state_dict(state_dict, "core.models.dit.DiT", prefix)
    old, new = prefix + "input.embedding.embedding", prefix + "input.views.0.representation.embedding"
    return {new if key == old else key: value for key, value in state.items()}


def mixture_step(cfg, size):
    """A training step at which the stratified block-size mixture picks ``size``."""
    mixture = instantiate(cfg.block_diffusion.training)
    return next(step for step in range(10_000) if mixture.sample(step=step) == size)


TRAINING = [
    ("uniform_diffusion", {}),
    ("udlm", {}),
    ("uniform_zero_time", {}),
    ("uniform_sc", {}),
    ("blockgen_uniform", {}),
    ("tinygsm_blockgen_uniform", {}),
    ("tinygsm_uniform_sc", {}),
    ("duo", {"step": 0}),  # Relaxation before the curriculum ramp.
    ("duo", {"step": 2}),  # Inside the ramp.
    ("duo", {"step": 3}),  # Inclusive final relaxation step.
    ("duo", {"step": 4}),  # Plain uniform loss after end_step.
    ("duo_sc", {"step": 2}),
    ("duo_sc", {"step": 4}),
]


@pytest.mark.parametrize("variant,options", TRAINING)
def test_training_is_identical(variant, options, duo_table):
    cfg, reference, current = pair(variant, duo_table)
    assert_training_identical(reference, current, *batch(cfg), **options)


@pytest.mark.parametrize("size", [1, 4])
def test_block_mixture_training_is_identical(size, duo_table):
    # Size 1 trains at pure noise with the per-call CE objective and fixed
    # time sampler; size 4 inherits the uniform objective.
    cfg, reference, current = pair("blockgen_uniform_mixture", duo_table)
    assert_training_identical(
        reference, current, *batch(cfg), step=mixture_step(cfg, size)
    )


@pytest.mark.parametrize("variant", ["uniform_diffusion", "duo"])
def test_evaluation_loss_is_identical(variant, duo_table):
    cfg, reference, current = pair(variant, duo_table)
    tokens, mask = batch(cfg)
    values = []
    for algorithm in (reference, current):
        algorithm.eval()
        torch.manual_seed(7)
        with torch.no_grad():
            values.append(algorithm.loss(tokens, mask, training=False)["loss"])
    assert torch.equal(*values)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_old_checkpoints_load_strictly(variant, duo_table):
    _, reference, _ = pair(variant, duo_table)
    # A differently seeded current model must take every tensor from the old
    # standalone checkpoint, as when evaluating an existing run.
    _, _, current = pair(variant, duo_table, seed=123)
    current.load_state_dict(converted(reference.state_dict()), strict=True)
    old, new = list(reference.parameters()), list(current.parameters())
    assert len(old) == len(new)
    assert all(torch.equal(a, b) for a, b in zip(old, new))


SAMPLERS = [
    {"_target_": "core.samplers.uniform_ancestral.UniformAncestralSampler", "final_denoise": "ancestral"},
    {"_target_": "core.samplers.uniform_ancestral.UniformAncestralSampler", "final_denoise": "greedy"},
    {
        "_target_": "core.samplers.uniform_ancestral.UniformAncestralSampler",
        "final_denoise": "ancestral",
        "use_float64": False,
    },
    {"_target_": "core.samplers.uniform_entropy_bounded.UniformEntropyBoundedSampler", "entropy_bound": 0.5},
    {
        "_target_": "core.samplers.uniform_entropy_bounded.UniformEntropyBoundedSampler",
        "entropy_bound": 0.5,
        "freeze": True,
        "use_float64": False,
    },
]
SAMPLING_VARIANTS = [
    "uniform_diffusion",
    "uniform_zero_time",
    "uniform_sc",
    "blockgen_uniform",
    "tinygsm_blockgen_uniform",
    "tinygsm_uniform_sc",
    "duo",
    "duo_sc",
]


@pytest.mark.parametrize("variant", SAMPLING_VARIANTS)
def test_sampling_is_identical(variant, duo_table):
    cfg, reference, current = pair(variant, duo_table)
    # A trained-looking output layer, so predictions are not uniform.
    with torch.no_grad():
        for algorithm in (reference, current):
            torch.manual_seed(5)
            torch.nn.init.normal_(algorithm.model.output_layer.linear.weight, std=0.5)
    tokens, _ = batch(cfg)
    prompt = tokens[:3]
    fixed = torch.zeros_like(prompt, dtype=torch.bool)
    fixed[:, :6] = True
    fixed[1] = True  # A fully fixed sequence.
    reference.eval()
    current.eval()
    for options in SAMPLERS:
        sampler = instantiate({"steps": 8, "epsilon": 1e-3, **options})
        torch.manual_seed(11)
        expected, expected_info = reference.generate(sampler, prompt, fixed)
        torch.manual_seed(11)
        actual, actual_info = current.generate(sampler, prompt, fixed)
        assert torch.equal(expected, actual), options
        assert expected_info["nfe_per_sample"] == actual_info["nfe_per_sample"], options
        assert expected_info["nfe"] == actual_info["nfe"], options

"""GaussianDiffusion on the view path matches the frozen standalone class.

Every Gaussian preset is built twice from the same config and seed: the frozen
class (core/tests/reference_algorithms/gaussian_diffusion.py) with the frozen
samplers (core/tests/reference_samplers/{continuous,gaussian_ddpm}.py), and the
current class with the current samplers. Initialization, loss, gradients,
schedule / time-sampler feedback, samples and NFE are compared with
torch.equal (they are exact on CPU).

ContinuousDiT held bare root parameters and the standalone class registered
the representation after the time sampler and objective, so parameters are
paired by shape and value (``ordered=False``) and old state dicts are renamed
(``convert``) before a strict load.
"""

from hydra.utils import instantiate
import pytest
import torch

from core.models.legacy import convert_state_dict
from core.tests.view_equivalence import assert_training_identical, batch, build

REFERENCE = "core.tests.reference_algorithms.gaussian_diffusion.GaussianDiffusion"
CONTINUOUS_DIT = "core.models.continuous_dit.ContinuousDiT"

# Every Gaussian preset; True marks the presets with self-conditioning on.
PRESETS = {
    "candi_embedding_pure_geometric_rank": False,
    "candi_embedding_pure_linear": False,
    "candi_embedding_pure_linear_rank": False,
    "candi_rank_pure_geometric": False,
    "candi_rank_pure_linear": False,
    "cdcd": False,
    "cfm_gaussian": False,
    "cont_diff_onehot": False,
    "embedding_flow_matching": False,
    "embedding_ve": False,
    "flm_gaussian": False,
    "gaussian_base": False,
    "gaussian_diffusion_embedding_sudoku": False,
    "gaussian_diffusion_sudoku": False,
    "langflow": True,
    "onehot_gaussian": False,
    "onehot_gaussian_adaptive": False,
    "semicat_gaussian": False,
    "tinygsm_gaussian_matched_blurring_sc_adaptive": True,
    "tinygsm_gaussian_sc_adaptive": True,
}

# Option variants beyond the presets' defaults.
VARIANTS = {
    "gaussian_base+sc": ("gaussian_base", {"self_conditioning.enabled": True}),
    "embedding+sc_hidden": (
        "candi_embedding_pure_linear",
        {"self_conditioning.enabled": True, "self_conditioning.source": "hidden"},
    ),
    "langflow-sc": ("langflow", {"self_conditioning.enabled": False}),
    # Context-weighted objectives read the training time and path coefficients.
    "power_time": (
        "gaussian_base",
        {"objective.weighting": {"_target_": "core.objectives.weighting.PowerLossWeighting", "coordinate": "time", "exponent": 1.0}},
    ),
    "power_sigma": (
        "embedding_ve",
        {"objective.weighting": {"_target_": "core.objectives.weighting.PowerLossWeighting", "coordinate": "sigma", "exponent": -1.0}},
    ),
    "time_input": ("gaussian_base", {"+model.time_input": True}),
    "linear_unit_embedding": (
        "candi_embedding_pure_linear",
        {
            "+model.input_architecture": "linear",
            "model.input_preconditioning": "unit_embedding",
            "+model.output_architecture": "adaln",
            "model.time_conditioning": "negative_log_snr",
            "+model.conditioning_scale": "unit_embedding",
            "self_conditioning.enabled": True,
        },
    ),
    "sc_both_gradients": (
        "gaussian_base",
        {"self_conditioning.enabled": True, "self_conditioning.gradient_mode": "both"},
    ),
    "sc_mixed_gradients": (
        "gaussian_base",
        {"self_conditioning.enabled": True, "self_conditioning.gradient_mode": "mixed"},
    ),
}
CASES = {**{name: (name, {}) for name in PRESETS}, **VARIANTS}


def pair(case, seed=0):
    experiment, overrides = CASES[case]
    cfg, reference = build(experiment, REFERENCE, overrides=overrides, seed=seed)
    _, current = build(experiment, None, overrides=overrides, seed=seed)
    return cfg, reference, current


def convert(state_dict):
    """A standalone GaussianDiffusion state dict under the view-path names."""
    converted = {}
    for key, value in convert_state_dict(state_dict, CONTINUOUS_DIT, "model.").items():
        if key.split(".", 1)[0] in ("process", "schedule", "representation"):
            key = "own_views.0." + key
        converted[key] = value
    return converted


def _tensors(component):
    """Every tensor state of a schedule or time sampler, by attribute path."""
    found, stack, seen = {}, [("", component)], set()
    while stack:
        prefix, value = stack.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if torch.is_tensor(value):
            found[prefix] = value
        elif isinstance(value, (list, tuple)):
            stack.extend((f"{prefix}[{i}]", item) for i, item in enumerate(value))
        elif isinstance(value, dict):
            stack.extend((f"{prefix}.{k}", item) for k, item in value.items())
        elif hasattr(value, "__dict__") and type(value).__module__.startswith("core."):
            stack.extend((f"{prefix}.{k}", item) for k, item in vars(value).items())
        elif isinstance(value, (int, float, bool)):
            found[prefix] = torch.tensor(value)
    return found


def assert_same_feedback(reference, current):
    for name in ("schedule", "time_sampler"):
        old, new = _tensors(getattr(reference, name)), _tensors(getattr(current, name))
        assert old.keys() == new.keys(), name
        for key in old:
            assert torch.equal(old[key], new[key]), (name, key)


@pytest.mark.parametrize("case", sorted(CASES))
def test_initialization_and_checkpoint_names(case):
    _, reference, current = pair(case)
    state = convert(reference.state_dict())
    assert state.keys() == current.state_dict().keys()
    for key, value in current.state_dict().items():
        assert value.shape == state[key].shape and torch.equal(value, state[key]), key
    # An old checkpoint loads strictly into a differently seeded current model.
    _, other = build(CASES[case][0], None, overrides=CASES[case][1], seed=123)
    other.load_state_dict(state, strict=True)
    for key, value in other.state_dict().items():
        assert torch.equal(value, state[key]), key


@pytest.mark.parametrize("case", sorted(CASES))
def test_training_and_feedback_are_identical(case):
    cfg, reference, current = pair(case)
    tokens, mask = batch(cfg)
    for step in range(2):
        for algorithm in (reference, current):
            algorithm.zero_grad(set_to_none=True)
        assert_training_identical(
            reference, current, tokens, mask, seed=7 + step, ordered=False, step=step
        )
        assert_same_feedback(reference, current)


def test_per_call_time_sampler_and_objective_overrides():
    cfg, reference, current = pair("gaussian_base")
    tokens, mask = batch(cfg)
    options = {
        "time_sampler": instantiate({"_target_": "core.time_sampling.uniform.UniformTimeSampler", "epsilon": 0.2}),
        "objective": instantiate(
            {
                "_target_": "core.objectives.cross_entropy.CrossEntropy",
                "label_smoothing": 0.1,
            }
        ),
    }
    assert_training_identical(reference, current, tokens, mask, ordered=False, **options)


def _euler(module, **options):
    return {"_target_": f"{module}.EulerSampler", "steps": 4, **options}


CONTINUOUS = ("core.tests.reference_samplers.continuous", "core.samplers")


def _samplers(sc):
    """(reference, current) sampler configs covering every preset setting."""
    ref, cur = CONTINUOUS
    pairs = [
        (_euler(ref), _euler(f"{cur}.euler")),
        (_euler(ref, decoding="representation"), _euler(f"{cur}.euler", decoding="representation")),
        (
            {"_target_": f"{ref}.HeunSampler", "steps": 3},
            {"_target_": f"{cur}.heun.HeunSampler", "steps": 3},
        ),
    ]
    if not sc:  # The adaptive ODE sampler rejects self-conditioning.
        pairs.append(
            (
                {"_target_": f"{ref}.ODESampler", "rtol": 0.05, "atol": 0.05, "max_nfe": 2000},
                {"_target_": f"{cur}.ode.ODESampler", "rtol": 0.05, "atol": 0.05, "max_nfe": 2000},
            )
        )
    ddpm = [
        {"prior_mode": "standard_normal", "prompt_mode": "noisy", "self_conditioning_mode": "previous"},
        {"prior_mode": "schedule", "prompt_mode": "clean", "self_conditioning_mode": "previous"},
        {"prior_mode": "standard_normal", "prompt_mode": "clean", "epsilon": 0.0},
        {"prompt_mode": "noisy", "self_conditioning_mode": "bootstrap", "final_transition_noise": True},
    ]
    for options in ddpm:
        options = {"steps": 4, "epsilon": 1e-3, **options}
        pairs.append(
            (
                {"_target_": "core.tests.reference_samplers.gaussian_ddpm.GaussianDDPMSampler", **options},
                {"_target_": "core.samplers.gaussian_ddpm.GaussianDDPMSampler", **options},
            )
        )
    return pairs


def assert_sampling_pairs_identical(reference, current, prompt, fixed, pairs, seed=11):
    reference.eval()
    current.eval()
    for old, new in pairs:
        torch.manual_seed(seed)
        expected, expected_info = reference.generate(instantiate(old), prompt, fixed)
        torch.manual_seed(seed)
        actual, actual_info = current.generate(instantiate(new), prompt, fixed)
        assert torch.equal(expected, actual), new
        assert expected_info["nfe_per_sample"] == actual_info["nfe_per_sample"], new


SAMPLING_CASES = sorted(
    {
        "gaussian_base",
        "gaussian_base+sc",
        "gaussian_diffusion_embedding_sudoku",
        "candi_embedding_pure_linear",
        "embedding+sc_hidden",
        "flm_gaussian",
        "cdcd",
        "onehot_gaussian_adaptive",
        "langflow",
        "linear_unit_embedding",
        "tinygsm_gaussian_sc_adaptive",
        "tinygsm_gaussian_matched_blurring_sc_adaptive",
    }
)


@pytest.mark.parametrize("case", SAMPLING_CASES)
def test_sampling_is_identical(case):
    cfg, reference, current = pair(case)
    tokens, _ = batch(cfg)
    prompt = tokens[:3]
    fixed = torch.zeros_like(prompt, dtype=torch.bool)
    fixed[:, :6] = True
    sc = cfg.self_conditioning.enabled
    assert_sampling_pairs_identical(reference, current, prompt, fixed, _samplers(sc))


@pytest.mark.parametrize("case", ["gaussian_base", "tinygsm_gaussian_matched_blurring_sc_adaptive"])
def test_preset_sampler_is_identical(case):
    cfg, reference, current = pair(case)
    tokens, _ = batch(cfg)
    prompt = tokens[:2]
    fixed = torch.zeros_like(prompt, dtype=torch.bool)
    fixed[:, :6] = True
    sampler = {**cfg.sampler, "steps": 5}
    old = {**sampler, "_target_": sampler["_target_"].replace(
        "core.samplers.euler", "core.tests.reference_samplers.continuous"
    ).replace("core.samplers.gaussian_ddpm", "core.tests.reference_samplers.gaussian_ddpm")}
    assert_sampling_pairs_identical(reference, current, prompt, fixed, [(old, sampler)])

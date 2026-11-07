"""The legacy network classes are presets of ModularDiT, bit for bit."""

from hydra.utils import instantiate
import pytest
import torch

from core.models.legacy import convert_parameter_order, convert_state_dict, modular_config
from core.modifiers.hybrid.views import ViewPoint
from core.noise_schedules.gaussian_path import GaussianPath

BASE = {"vocab_size": 11, "hidden_size": 16, "cond_dim": 8, "n_blocks": 2, "n_heads": 2, "dropout": 0.0}
DIT = {"_target_": "core.models.dit.DiT", **BASE}
SIMPLEX = {"_target_": "core.models.simplex.SimplexDiT", **BASE, "adaLN": True}
SPHERE = {"_target_": "core.models.spherical.SphereDiT", **BASE, "state_dim": 12}
CONTINUOUS = {"_target_": "core.models.continuous_dit.ContinuousDiT", **BASE, "state_dim": 12}

CASES = {
    "dit": DIT,
    "dit_adaln": {**DIT, "adaLN": True},
    "dit_sc_mean": {**DIT, "adaLN": True, "self_conditioning": True},
    "dit_sc_hidden": {**DIT, "self_conditioning": True, "self_conditioning_source": "hidden"},
    "dit_tied": {**DIT, "tie_word_embeddings": True},
    "simplex_expectation": SIMPLEX,
    "simplex_argmax": {**SIMPLEX, "input_mode": "argmax"},
    "sphere": SPHERE,
    "sphere_temperature_sc": {**SPHERE, "learn_temperature_scaling": True, "self_conditioning": True},
    "continuous": CONTINUOUS,
    "continuous_linear_adaln": {
        **CONTINUOUS,
        "input_architecture": "linear",
        "input_preconditioning": "none",
        "time_conditioning": "time",
        "output_architecture": "adaln",
    },
    "continuous_time_input_log_snr": {**CONTINUOUS, "time_input": True, "time_conditioning": "log_snr"},
    "continuous_sc_unit_scale": {
        **CONTINUOUS,
        "self_conditioning": True,
        "conditioning_scale": "unit_embedding",
        "conditioning_initialization": "default",
    },
}


def inputs(name, config):
    torch.manual_seed(5)
    batch, length = 3, 6
    time = torch.rand(batch) * 0.9 + 0.05
    alpha, sigma = torch.rand(batch) + 0.1, torch.rand(batch) + 0.1
    if name.startswith("dit"):
        state = torch.randint(0, config["vocab_size"], (batch, length))
        legacy = (state, time)
    elif name.startswith("simplex"):
        state = torch.randn(batch, length, config["vocab_size"]).log_softmax(-1)
        legacy = (state, time)
    elif name.startswith("sphere"):
        state = torch.randn(batch, length, config["state_dim"])
        legacy = (state, time)
    else:
        state = torch.randn(batch, length, config["state_dim"])
        legacy = (state, time, sigma, alpha)
    point = ViewPoint(time, GaussianPath(alpha, sigma, torch.zeros_like(alpha), torch.zeros_like(alpha)))
    modular = ((state,), (point,), time)
    feedback = None
    if config.get("self_conditioning"):
        width = config["state_dim"] if "state_dim" in config else config["hidden_size"]
        feedback = torch.randn(batch, length, width)
    return legacy, modular, feedback


@pytest.mark.parametrize("name", sorted(CASES))
def test_legacy_network_is_a_modular_preset(name):
    config = CASES[name]
    torch.manual_seed(0)
    legacy = instantiate(config)
    torch.manual_seed(0)
    modular = instantiate(modular_config(config))
    # Same initialization from the same seed (same creation order). Listing
    # order can differ, so compare by converted name and convert EMA order.
    converted = convert_state_dict(legacy.state_dict(), config["_target_"])
    assert set(converted) == set(modular.state_dict())
    assert all(torch.equal(v, modular.state_dict()[k]) for k, v in converted.items())
    reordered = convert_parameter_order(legacy, modular, list(legacy.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(reordered, modular.parameters()))
    # Converted weights load strictly into a differently seeded network.
    torch.manual_seed(99)
    loaded = instantiate(modular_config(config))
    loaded.load_state_dict(
        convert_state_dict(legacy.state_dict(), config["_target_"]), strict=True
    )
    legacy.eval()
    loaded.eval()
    legacy_args, modular_args, feedback = inputs(name, config)
    with torch.no_grad():
        expected = legacy(*legacy_args, self_condition=feedback, return_features=True)
        actual = loaded(*modular_args, self_condition=feedback, return_features=True)
    assert torch.equal(expected.logits, actual.logits)
    assert torch.equal(expected.hidden, actual.hidden)

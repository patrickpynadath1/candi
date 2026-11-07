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


def view(process, representation, update, schedule=None):
    return {
        "_target_": "core.modifiers.hybrid.views.DiffusionView",
        "process": process,
        "schedule": schedule or {"_target_": "core.noise_schedules.log_linear.LogLinear", "epsilon": 0.001},
        "representation": representation,
        "update": {"_target_": f"core.samplers.hybrid_updates.{update}"},
    }


MASKED = view(
    {"_target_": "core.processes.absorbing.AbsorbingProcess", "mask_index": 11},
    {"_target_": "core.representations.noisy_inputs.TokenRepresentation", "vocab_size": 12, "hidden_size": 32},
    "DiscretePosteriorUpdate",
)
GAUSSIAN = view(
    {"_target_": "core.processes.gaussian.GaussianProcess"},
    {
        "_target_": "core.representations.noisy_inputs.ProjectedRepresentation",
        "state_dim": 11,
        "hidden_size": 32,
        "state_representation": {"_target_": "core.representations.one_hot.OneHotRepresentation", "vocab_size": 11},
    },
    "GaussianPosteriorUpdate",
    schedule={"_target_": "core.noise_schedules.gaussian_rank.OneHotRankSchedule", "vocab_size": 11},
)
BACKBONE = {"_target_": "core.models.dit.DiT", "vocab_size": 11, "hidden_size": 32, "cond_dim": 16,
            "n_blocks": 1, "n_heads": 2, "dropout": 0.0, "adaLN": True}
HYBRID = "core.modifiers.hybrid.model.HybridModel"
HYBRID_CASES = {
    "masked": {"_target_": HYBRID, "backbone": BACKBONE, "views": [MASKED], "output_vocab_size": 11},
    "gaussian_only": {"_target_": HYBRID, "backbone": BACKBONE, "views": [GAUSSIAN]},
    # negative_log_keep reads the first view's keep rate, so masking goes first.
    "masked_gaussian_sc": {
        "_target_": HYBRID, "backbone": BACKBONE, "views": [MASKED, GAUSSIAN], "output_vocab_size": 11,
        "self_conditioning": True, "time_conditioning": "negative_log_keep",
    },
    "gated_fusion_hidden_sc": {
        "_target_": HYBRID, "backbone": BACKBONE, "views": [MASKED, GAUSSIAN], "output_vocab_size": 11,
        "self_conditioning": True, "self_conditioning_source": "hidden", "time_conditioning": "log_one_minus_time",
        "fusion": {"_target_": "core.modifiers.hybrid.fusion.GatedFusion", "mask_index": 11,
                   "weights": [{"masked": 1, "revealed": 2}, {"masked": 1, "revealed": 0}]},
    },
}


@pytest.mark.parametrize("name", sorted(HYBRID_CASES))
def test_legacy_hybrid_model_is_a_modular_preset(name):
    from core.models.legacy import convert_hybrid_state_dict
    from core.modifiers.hybrid.views import point

    config = HYBRID_CASES[name]
    torch.manual_seed(0)
    legacy = instantiate(config)
    torch.manual_seed(99)
    modular = instantiate(modular_config(config))
    modular.load_state_dict(convert_hybrid_state_dict(legacy.state_dict()), strict=True)
    legacy.eval()
    modular.eval()
    torch.manual_seed(3)
    tokens = torch.randint(0, 11, (2, 6))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, 0] = False
    time = torch.tensor([0.3, 0.8])
    points = tuple(point(v, time, corruption_mask=mask) for v in legacy.views)
    states = tuple(
        v.process.corrupt_state(v.representation.encode(tokens), p.coefficients, mask)
        for v, p in zip(legacy.views, points)
    )
    feedback = torch.randn(2, 6, 32) if config.get("self_conditioning") else None
    with torch.no_grad():
        expected = legacy(states, points, time, self_condition=feedback, return_features=True)
        actual = modular(states, points, time, self_condition=feedback, return_features=True)
        assert torch.equal(expected.logits, actual.logits)
        assert torch.equal(expected.hidden, actual.hidden)
        probabilities = torch.rand(2, 6, modular.output_width).softmax(-1)
        assert torch.equal(legacy.clean_embedding(probabilities), modular.clean_embedding(probabilities))
        old_hidden, old_condition = legacy.clean_state_hidden(probabilities)
        new_hidden, new_condition = modular.clean_hidden(probabilities)
        assert torch.equal(old_hidden, new_hidden) and torch.equal(old_condition, new_condition)
        old_table, new_table = legacy.feedback_embedding_table(), modular.feedback_embedding_table()
        assert (old_table is None) == (new_table is None)
        if old_table is not None:
            assert torch.equal(old_table, new_table)

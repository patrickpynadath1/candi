"""Fusion contracts and numerical parity with the original CANDI assembly."""

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from core.modifiers.hybrid.fusion import GatedFusion
from core.modifiers.hybrid.checkpoint import upgrade_config
from core.representations.noisy_inputs import CANDIRepresentation, TokenRepresentation


def original_candi(tokens, state, sigma, table, discrete_scale, continuous_scale):
    normalized = F.normalize(table.float(), p=2, dim=-1)
    clean = F.embedding(tokens.clamp_max(len(table) - 2), normalized[:-1])
    mask = table.float()[-1]
    mask = mask / mask.norm().clamp_min(1e-12)
    continuous = (state.float() @ normalized[:-1]).to(state.dtype)
    noisy = (
        discrete_scale * mask
        + continuous_scale
        * continuous
        * (1 / (1 + sigma.float().square()).sqrt())[:, None, None]
    )
    return torch.where(tokens.ne(len(table) - 1)[..., None], clean, noisy)


@pytest.mark.parametrize("layout", ["mixed", "masked", "revealed"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_candi_outputs_and_gradients_match_reference(layout, dtype):
    torch.manual_seed(41)
    discrete = TokenRepresentation(6, 8, scale=0.5, normalize=True)
    continuous = CANDIRepresentation(5, scale=0.5)
    continuous.bind_embedding(discrete.embedding)
    tokens = torch.tensor([[5, 1, 2], [0, 5, 5]])
    if layout == "masked":
        tokens.fill_(5)
    elif layout == "revealed":
        tokens.clamp_max_(4)
    state = torch.randn(2, 3, 5).to(dtype).requires_grad_()
    sigma = torch.tensor([0.2, 3.0], requires_grad=True)
    points = [None, SimpleNamespace(coefficients=SimpleNamespace(sigma=sigma))]
    embeddings = [
        discrete.to_model_embedding(tokens, None),
        continuous.to_model_embedding(state, points[1]),
    ]
    fusion = GatedFusion(
        5, [{"masked": 1, "revealed": 2}, {"masked": 1, "revealed": 0}]
    )
    actual = fusion([tokens, state], points, embeddings, None)
    expected = original_candi(tokens, state, sigma, discrete.embedding, 0.5, 0.5)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    cotangent = torch.randn_like(actual)
    variables = (discrete.embedding, state, sigma)
    actual_grads = torch.autograd.grad(actual, variables, cotangent, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, variables, cotangent)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_reordered_gate_and_three_views():
    tokens = torch.tensor([[3, 1]])
    embeddings = [
        torch.full((1, 2, 4), value, requires_grad=True) for value in (2.0, 3.0, 5.0)
    ]
    fusion = GatedFusion(
        3,
        [
            {"masked": 1, "revealed": 0},
            {"masked": 1, "revealed": 1},
            {"masked": 0, "revealed": 2},
        ],
        discrete_view=1,
    )
    actual = fusion([torch.empty(0), tokens, torch.empty(0)], None, embeddings, None)
    torch.testing.assert_close(actual, torch.tensor([[[5.0] * 4, [13.0] * 4]]))
    actual.sum().backward()
    torch.testing.assert_close(
        embeddings[0].grad, torch.tensor([[[1.0] * 4, [0.0] * 4]])
    )


def test_excluded_nonfinite_view_does_not_leak():
    tokens = torch.tensor([[3, 1]])
    fusion = GatedFusion(
        3, [{"masked": 1, "revealed": 1}, {"masked": 1, "revealed": 0}]
    )
    actual = fusion(
        [tokens, torch.empty(0)],
        None,
        [torch.ones(1, 2, 1), torch.tensor([[[2.0], [float("nan")]]])],
        None,
    )
    torch.testing.assert_close(actual, torch.tensor([[[3.0], [1.0]]]))


@pytest.mark.parametrize(
    "weights,index",
    [
        ([], 0),
        ([{"masked": 1}], 0),
        ([{"masked": 1, "revealed": float("inf")}], 0),
        ([{"masked": 1, "revealed": 0}], 1),
    ],
)
def test_invalid_configuration(weights, index):
    with pytest.raises(ValueError):
        GatedFusion(3, weights, discrete_view=index)


def test_invalid_state_or_embedding_layout():
    fusion = GatedFusion(3, [{"masked": 1, "revealed": 0}])
    with pytest.raises(ValueError, match="token IDs"):
        fusion([torch.zeros(1, 2)], None, [torch.zeros(1, 2, 4)], None)
    with pytest.raises(ValueError, match="layout"):
        fusion(
            [torch.zeros(1, 2, dtype=torch.long)], None, [torch.zeros(1, 3, 4)], None
        )
    with pytest.raises(ValueError, match="per fusion weight"):
        fusion([], None, [], None)


@pytest.mark.parametrize(
    "target",
    ["core.models.hybrid.CANDIFusion", "core.modifiers.hybrid.model.CANDIFusion"],
)
def test_saved_config_migration(target):
    old = {
        "_target_": "core.modifiers.hybrid.model.HybridModel",
        "fusion": {"_target_": target, "mask_index": 5},
        "views": [{"representation": {"scale": 0.25}}, {}],
    }
    new = upgrade_config(old)
    assert new["fusion"]["_target_"] == "core.modifiers.hybrid.fusion.GatedFusion"
    assert new["fusion"]["weights"] == [
        {"masked": 1, "revealed": 4},
        {"masked": 1, "revealed": 0},
    ]
    assert upgrade_config(new) == new
    assert "weights" not in old["fusion"]


def test_legacy_adapter_migration_preserves_scale_interpolation_and_normalization():
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    old = {
        "input_scale": 0.25,
        "model": {
            "_target_": "core.models.hybrid.HybridModel",
            "branches": [
                {
                    "_target_": "core.hybrid.branches.DiscreteBranch",
                    "representation": {"unused": True},
                    "adapter": {
                        "_target_": "core.models.hybrid.TokenEmbeddingAdapter",
                        "scale": "${input_scale}",
                        "normalize": False,
                    },
                },
                {},
            ],
            "fusion": {
                "_target_": "core.models.hybrid.CANDIFusion",
                "discrete_branch": 0,
                "mask_index": 5,
            },
        },
    }
    new = OmegaConf.create(upgrade_config(old))
    fusion = instantiate(new.model.fusion)
    assert fusion.weights == [(1.0, 4.0), (1.0, 0.0)]
    assert new.model.views[0].representation.normalize
    assert not old["model"]["branches"][0]["adapter"]["normalize"]

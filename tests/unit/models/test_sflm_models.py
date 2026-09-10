from types import SimpleNamespace

import pytest
import torch

from models.flm_dit import FLMDIT
from models.sphere_arch import SphereArch
from models.sphere_dit import SphereDiT


def _config(**overrides):
    model = dict(
        hidden_size=8,
        cond_dim=8,
        n_heads=2,
        n_blocks=1,
        dropout=0.0,
        eps=1e-6,
        init="ngpt",
        learn_temperature_scaling=False,
        mlp_type="gelu",
        use_time_alpha=True,
    )
    model.update(overrides)
    return SimpleNamespace(
        model=SimpleNamespace(**model),
        algo=SimpleNamespace(double_temb=False, learnable_loss_weighting=False),
    )


@pytest.mark.parametrize("backbone", [FLMDIT, SphereDiT, SphereArch])
def test_sflm_backbones_have_finite_token_logits(backbone):
    model = backbone(_config(), vocab_size=7)
    if backbone is FLMDIT:
        inputs = torch.randn(2, 4, 7)
    else:
        inputs = torch.nn.functional.normalize(torch.randn(2, 4, 8), dim=-1)
    output = model(inputs, torch.rand(2))
    assert output.shape == (2, 4, 7)
    assert torch.isfinite(output).all()


def test_flm_embedding_does_not_softmax_continuous_input():
    model = FLMDIT(_config(), vocab_size=3)
    with torch.no_grad():
        model.vocab_embed.embedding.copy_(torch.arange(24).reshape(3, 8))
    state = torch.tensor([[[2.0, -1.0, 0.5]]])
    expected = torch.einsum("blv,vd->bld", state, model.vocab_embed.embedding)
    assert torch.equal(model.vocab_embed(state), expected)


def test_spherical_embeddings_are_normalized():
    for backbone in (SphereDiT, SphereArch):
        model = backbone(_config(), vocab_size=7)
        embeddings = model.get_sphere_embeddings(torch.arange(7))
        assert torch.allclose(embeddings.norm(dim=-1), torch.ones(7), atol=1e-6)


def test_sphere_dit_projects_configurable_spherical_dimension_to_hidden_size():
    model = SphereDiT(
        _config(input_embedding_dim=16, hidden_size=32, n_heads=4),
        vocab_size=7,
    )
    tokens = torch.tensor([[0, 1, 2], [3, 4, 5]])
    state = model.get_sphere_embeddings(tokens)
    logits = model(state, torch.tensor([0.2, 0.8]))

    assert state.shape == (2, 3, 16)
    assert model.sphere_embed.weight.shape == (7, 16)
    assert model.input_projection.in_features == 16
    assert model.input_projection.out_features == 32
    assert logits.shape == (2, 3, 7)
    assert torch.isfinite(logits).all()

    with pytest.raises(ValueError, match="16.*spherical embeddings"):
        model(torch.randn(2, 3, 32), torch.tensor([0.2, 0.8]))


@pytest.mark.parametrize("mlp_type", ["gelu", "swiglu"])
@pytest.mark.parametrize("time_alpha", [True, False])
def test_sphere_arch_projection_and_trainable_mlp_scale(mlp_type, time_alpha):
    model = SphereArch(
        _config(mlp_type=mlp_type, use_time_alpha=time_alpha, use_time_token=True),
        vocab_size=7,
    )
    block = model.blocks[0]
    time_before = {k: v.clone() for k, v in model.time.state_dict().items()}
    model.renormalize_weights()
    for layer in (block.q, block.k, block.v, block.fc):
        assert torch.allclose(
            layer.weight.norm(dim=1), torch.ones(layer.weight.shape[0]), atol=1e-6
        )
    for layer in (block.o, block.proj):
        assert torch.allclose(
            layer.weight.norm(dim=0), torch.ones(layer.weight.shape[1]), atol=1e-6
        )
    for key, value in model.time.state_dict().items():
        assert torch.equal(value, time_before[key])
    x = torch.nn.functional.normalize(torch.randn(2, 4, 8), dim=-1)
    model(x, torch.rand(2)).square().mean().backward()
    assert torch.isfinite(block.mlp_scale.grad).all()
    assert block.mlp_scale.grad.abs().sum() > 0

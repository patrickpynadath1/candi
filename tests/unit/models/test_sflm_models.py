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

import pytest
import torch
from torch import nn
from core.models.dit import DiT
from core.training.procedures.maps import (
    LinearInterpolation,
    linear_transport,
    MeanDenoiserMap,
    EndpointMap,
    SemigroupTarget,
    DirectionalEndpointTarget,
    LearnedMapStep,
    EndpointRefinementStep,
    iterate_maps,
)


def test_interpolation_coupling_derivative_and_context():
    tokens = torch.tensor([[0, 1]])
    noise = torch.tensor([[[2.0, -1.0], [-2.0, 3.0]]])
    path = LinearInterpolation()
    state, velocity = path.sample(tokens, 2, 0.4, noise, torch.tensor([[True, False]]))
    later, _ = path.sample(tokens, 2, 0.4 - 1e-3, noise, torch.tensor([[True, False]]))
    torch.testing.assert_close((later - state) / 1e-3, velocity, atol=1e-4, rtol=1e-4)
    assert torch.equal(state[:, 1], torch.tensor([[0.0, 1.0]]))
    assert torch.equal(velocity[:, 1], torch.zeros(1, 2))


def test_mean_denoiser_psd_matches_composed_transport():
    torch.manual_seed(4)
    state = torch.randn(2, 3, 4)
    s, m, d = (
        torch.tensor([0.9, 0.8]),
        torch.tensor([0.6, 0.3]),
        torch.tensor([0.2, 0.0]),
    )

    def predictor(x, source, destination):
        return (x * 0.7 + destination[:, None, None]).log_softmax(-1)

    target = SemigroupTarget()(predictor, state, s, m, d)
    composed = linear_transport(state, predictor(state, s, m).exp(), s, m)
    composed = linear_transport(composed, predictor(composed, m, d).exp(), m, d)
    direct = linear_transport(state, target, s, d)
    torch.testing.assert_close(direct, composed)
    torch.testing.assert_close(target.sum(-1), torch.ones(2, 3))
    assert not target.requires_grad
    torch.testing.assert_close(linear_transport(state, target, 0.0, 0.0), state)
    with pytest.raises(ValueError, match="PSD"):
        SemigroupTarget()(predictor, state, s, s, d)


def test_directional_target_matches_analytic_softmax_jvp():
    x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    v = torch.randn_like(x)

    def fn(z):
        return z.log_softmax(-1)

    target = DirectionalEndpointTarget()(fn, x, v)
    p = x.softmax(-1)
    tangent = p * (v - (p * v).sum(-1, keepdim=True))
    torch.testing.assert_close(target, (p + tangent).detach())
    assert not target.requires_grad
    loss = (p - target).square().sum()
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def backbone():
    return DiT(
        5, hidden_size=16, cond_dim=8, n_blocks=2, n_heads=2, dropout=0.0, adaLN=True
    )


@pytest.mark.parametrize("kind", ["map", "endpoint"])
def test_backbone_adapter_backward(kind):
    model = (
        MeanDenoiserMap(backbone(), 5) if kind == "map" else EndpointMap(backbone(), 5)
    )
    x = torch.randn(2, 3, 5)
    result = model(x, 0.8, 0.2) if kind == "map" else model(x)
    assert result.shape == x.shape
    torch.testing.assert_close(result.exp().sum(-1), torch.ones(2, 3))
    (-result[..., 0].mean()).backward()
    assert model.backbone.output_layer.linear.weight.grad is not None


def test_map_iteration_context_nfe_and_gradients():
    param = nn.Parameter(torch.randn(1, 1, 4))

    def predictor(x, s, d):
        return (x + param).log_softmax(-1)

    x = torch.randn(1, 2, 4)
    context = torch.nn.functional.one_hot(torch.tensor([[2, 1]]), 4).float()
    mask = torch.tensor([[True, False]])
    result, nfe = iterate_maps(
        predictor, x, [1.0, 0.5, 0.0], LearnedMapStep(), context, mask
    )
    assert nfe == 2
    torch.testing.assert_close(result[:, 0], context[:, 0])
    result[..., 0].sum().backward()
    assert param.grad.abs().sum() > 0

    def endpoint(x):
        return x.log_softmax(-1)

    refined, count = iterate_maps(
        endpoint, x, [1.0, 0.5, 0.0], EndpointRefinementStep()
    )
    torch.testing.assert_close(refined, x.softmax(-1).softmax(-1))
    assert count == 2


def test_endpoint_backbone_jvp_and_detachment():
    model = EndpointMap(backbone(), 5).eval()
    # Final projection starts at zero; use nonzero weights to exercise JVP.
    nn.init.normal_(model.backbone.output_layer.linear.weight, std=0.05)
    x = torch.randn(1, 3, 5)
    v = torch.randn_like(x)
    target = DirectionalEndpointTarget()(model, x, v)
    eps = 1e-3
    with torch.no_grad():
        value = model(x).exp()
        tangent = (model(x + eps * v).exp() - model(x - eps * v).exp()) / (2 * eps)
    torch.testing.assert_close(target, value + tangent, atol=2e-4, rtol=2e-3)
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("endpoint", [False, True])
def test_map_training_module_update_and_generation(endpoint):
    import lightning as L
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from core.training.module import TrainingModule
    from core.training.procedures.map_training import MapSampler

    config = OmegaConf.create(
        dict(
            mode="train",
            compile=False,
            algorithm=dict(
                _target_="core.training.procedures.map_training.MapAlgorithm",
                endpoint=endpoint,
                length=3,
                model=dict(
                    _target_="core.training.procedures.maps."
                    + ("EndpointMap" if endpoint else "MeanDenoiserMap"),
                    vocab_size=5,
                    backbone=dict(
                        _target_="core.models.dit.DiT",
                        vocab_size=5,
                        hidden_size=16,
                        cond_dim=8,
                        n_blocks=2,
                        n_heads=2,
                        dropout=0.0,
                        adaLN=True,
                    ),
                ),
            ),
            training=dict(ema=0.0, warmup_steps=0),
            optimizer=dict(
                lr=0.001, beta1=0.9, beta2=0.99, epsilon=1e-8, weight_decay=0.0
            ),
            loader=dict(global_batch_size=2),
        )
    )
    config.procedure = OmegaConf.load(
        "configs/procedure/beckmann.yaml"
        if endpoint
        else "configs/procedure/flow_map.yaml"
    )
    config.procedure.phases[0].states.warmup_steps = 0
    module = TrainingModule(config)
    batch = [
        dict(
            input_ids=torch.tensor([0, 1, 2]), loss_mask=torch.ones(3, dtype=torch.bool)
        )
        for _ in range(2)
    ]
    trainer = L.Trainer(
        max_steps=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    before = module.algorithm.model.backbone.output_layer.linear.weight.detach().clone()
    trainer.fit(module, train_dataloaders=DataLoader(batch, batch_size=2))
    assert not torch.equal(
        before, module.algorithm.model.backbone.output_layer.linear.weight
    )
    prompts = torch.tensor([[0, 1, 2]])
    mask = torch.tensor([[True, False, False]])
    policy = EndpointRefinementStep() if endpoint else LearnedMapStep()
    tokens, info = module.algorithm.generate(MapSampler(policy, steps=4), prompts, mask)
    assert tokens.shape == prompts.shape and tokens[0, 0] == 0
    assert info["nfe"] == 4


def test_semigroup_preserves_conditioning_for_second_prediction():
    state = torch.randn(1, 2, 3)
    prompt = torch.nn.functional.one_hot(torch.tensor([[1, 2]]), 3).float()
    mask = torch.tensor([[True, False]])
    calls = []

    def predictor(x, s, d):
        calls.append(x.clone())
        return x.log_softmax(-1)

    SemigroupTarget()(predictor, state, 0.9, 0.5, 0.1, prompt, mask)
    torch.testing.assert_close(calls[1][:, 0], prompt[:, 0])

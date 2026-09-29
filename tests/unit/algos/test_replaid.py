import torch
import pytest
from hydra import compose, initialize_config_dir

from algos import algo_registry
from dataloader import Text8Tokenizer
from pathlib import Path
from algos.replaid import PlaidGammaSchedule


CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")


def make_model(*extra_overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        config = compose(
            config_name="config",
            overrides=[
                "algo=replaid",
                "model=tiny",
                "data=ptb",
                "training.ema=0",
                "model.length=4",
                "model.hidden_size=8",
                "model.cond_dim=8",
                "model.n_blocks=1",
                "model.n_heads=2",
                "model.dropout=0",
                "sampling.predictor=replaid_ddpm",
                *extra_overrides,
            ],
        )
    return algo_registry.get("replaid")(config, Text8Tokenizer())


def test_schedule_is_monotone_and_vp_corruption_has_expected_endpoints():
    torch.manual_seed(1)
    model = make_model()
    times = torch.linspace(0, 1, 17)
    gamma, derivative = model._gamma(times)

    assert torch.all(gamma[1:] > gamma[:-1])
    assert torch.all(derivative > 0)
    assert torch.allclose(
        torch.sigmoid(-gamma) + torch.sigmoid(gamma), torch.ones_like(gamma)
    )


def test_replaid_loss_is_finite_and_updates_embeddings():
    torch.manual_seed(2)
    model = make_model()
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    result = model._loss(tokens, torch.ones_like(tokens), train_mode=True)
    result.loss.backward()

    gradient = model.backbone.token_embedding.grad
    assert torch.isfinite(result.loss)
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_output_prior_prefers_nearby_embedding():
    model = make_model()
    tokens = torch.tensor([[1, 2]])
    clean = model.backbone.embed_tokens(tokens)
    gamma = torch.full((1,), -8.0)
    log_probs = model.backbone(clean, gamma, output_prior=True)

    assert torch.equal(log_probs.argmax(-1), tokens)


def test_efficient_output_prior_matches_gaussian_logits():
    model = make_model()
    z_t = torch.randn(2, 3, model.config.algo.embedding_dim)
    gamma = torch.tensor([-1.0, 0.7])
    embeddings = model.backbone.normalized_embedding
    alpha = torch.sigmoid(-gamma).sqrt()
    sigma_sq = torch.sigmoid(gamma)
    full = -(
        z_t[:, :, None] - alpha[:, None, None, None] * embeddings[None, None]
    ).square().sum(-1) / (2 * sigma_sq[:, None, None])
    efficient = torch.einsum("ble,ve->blv", z_t, embeddings)
    efficient = efficient * (alpha / sigma_sq)[:, None, None]

    assert torch.allclose(
        full - full[..., :1], efficient - efficient[..., :1], atol=1e-5
    )


def test_replaid_generates_tokens_through_registered_ddpm_sampler():
    torch.manual_seed(3)
    model = make_model()
    model.eval()

    samples = model.generate_samples(num_samples=2, num_steps=2)

    assert samples.shape == (2, 4)
    assert samples.dtype == torch.long
    assert torch.all((0 <= samples) & (samples < model.vocab_size))


def test_self_conditioning_accepts_lower_precision_predictions(monkeypatch):
    model = make_model()
    original = model.denoise

    def lower_precision(*args, **kwargs):
        return original(*args, **kwargs).to(torch.bfloat16)

    monkeypatch.setattr(model, "denoise", lower_precision)
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    result = model._loss(tokens, torch.ones_like(tokens), train_mode=True)

    assert torch.isfinite(result.loss)


def test_nelbo_aggregation_remains_float64():
    model = make_model()
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    result = model._loss(tokens, torch.ones_like(tokens), train_mode=True)

    assert result.loss.dtype == torch.float64
    assert result.nlls.dtype == torch.float64


def test_validation_self_conditions_every_example(monkeypatch):
    model = make_model()
    original = model.denoise
    call_sizes = []

    def capture(*args, **kwargs):
        call_sizes.append(args[0].shape[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "denoise", capture)
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1], [1, 3, 2, 4]])

    model._loss(tokens, torch.ones_like(tokens), train_mode=False)

    assert call_sizes == [tokens.shape[0], tokens.shape[0]]


def test_fully_self_conditioned_rows_detach_embedding_likelihood_paths():
    torch.manual_seed(17)
    model = make_model("algo.self_condition_rate=1.0", "algo.output_prior=false")
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    result = model._loss(tokens, torch.ones_like(tokens), train_mode=True)
    result.loss.backward()

    embedding_gradient = model.backbone.token_embedding.grad
    assert embedding_gradient is None or torch.allclose(
        embedding_gradient, torch.zeros_like(embedding_gradient), atol=1e-7
    )
    assert torch.count_nonzero(model.backbone.output_layer.linear.weight.grad) > 0


def test_singleton_includes_reconstruction_and_diffusion(monkeypatch):
    model = make_model("algo.self_condition_rate=0", "algo.output_prior=false")
    tokens = torch.tensor([[1, 2, 3, 4]])
    monkeypatch.setattr(model, "sample_t", lambda count, device: torch.full((count,), 0.5, device=device))
    # A uniform predictor makes the expected objective independent of noise.
    monkeypatch.setattr(model, "denoise", lambda z, *args, **kwargs: torch.full(
        (*z.shape[:2], model.vocab_size), -torch.log(torch.tensor(float(model.vocab_size)))
    ))
    actual = model._loss(tokens, torch.ones_like(tokens)).loss
    clean = model.backbone.embed_tokens(tokens).double()
    prediction = model.backbone.normalized_embedding.mean(0).double()
    gamma, derivative = model._gamma(torch.tensor([0.5]))
    diffusion = 0.5 * derivative * torch.exp(-gamma) * (prediction - clean).square().sum(-1).mean()
    gamma1, _ = model._gamma(torch.ones(1))
    alpha_sq, sigma_sq = torch.sigmoid(-gamma1), torch.sigmoid(gamma1)
    prior = 0.5 * (alpha_sq * clean.square().sum(-1).mean() + model.config.algo.embedding_dim * (sigma_sq - 1 - sigma_sq.log()))
    expected = torch.log(torch.tensor(float(model.vocab_size))) + diffusion + prior
    assert torch.allclose(actual, expected.squeeze(), atol=1e-6)


def test_clean_context_is_preserved_during_training(monkeypatch):
    model = make_model("algo.clean_context=true", "algo.self_condition_rate=1")
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    mask = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
    states = []
    original = model.denoise

    def capture(z, *args, **kwargs):
        states.append(z.detach().clone())
        return original(z, *args, **kwargs)

    monkeypatch.setattr(model, "denoise", capture)
    loss = model._loss(tokens, mask, train_mode=True).loss
    loss.backward()
    clean = model.backbone.embed_tokens(tokens)
    # Bootstrap rows may be permuted; the actual denoising pass preserves order.
    assert torch.equal(states[-1][:, :2], clean[:, :2])
    assert not torch.equal(states[-1][:, 2:], clean[:, 2:])
    assert torch.isfinite(loss)


def test_disabled_self_conditioning_skips_validation_bootstrap(monkeypatch):
    model = make_model("algo.self_condition_rate=0")
    calls = []
    original = model.denoise

    def capture(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "denoise", capture)
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    model._loss(tokens, torch.ones_like(tokens))
    assert len(calls) == 1


def test_plaid_schedule_derivative_endpoints_and_curvature():
    schedule = PlaidGammaSchedule(-10, 5, 16).double()
    times = torch.linspace(0, 1, 101, dtype=torch.float64, requires_grad=True)
    gamma, analytic = schedule(times)
    autodiff = torch.autograd.grad(gamma.sum(), times, create_graph=True)[0]
    assert torch.allclose(analytic, autodiff, atol=1e-12)
    assert torch.all(analytic > 0)
    assert torch.allclose(gamma[[0, -1]], torch.tensor([-10., 5.], dtype=torch.float64))
    # A centered tanh has positive then negative curvature. The previous
    # convex-only basis could not represent this legitimate monotone shape.
    with torch.no_grad():
        schedule.bias.zero_()
    _, slope = schedule(times)
    assert slope[0] < slope[50] and slope[-1] < slope[50]


def test_schedule_hook_matches_second_moment_but_endpoints_match_mean_loss():
    model = make_model("algo.schedule_hidden_dim=16")
    t = torch.tensor([0.1, 0.4, 0.8], dtype=torch.float64)
    params = list(model.gamma_schedule.parameters())

    def losses(gamma, prime):
        # Differentiable stand-in denoiser: includes both schedule paths.
        return 0.5 * prime * torch.exp(-gamma) * (torch.sigmoid(gamma) + 0.2).square()

    gamma, prime, hook_weight = model._training_gamma(t, 0)
    hooked_loss = losses(gamma, prime)
    hook_weight[0] = 2 * hooked_loss.detach()
    actual = torch.autograd.grad(hooked_loss.mean(), params)
    gamma, prime = model._gamma(t)
    reference = losses(gamma, prime)
    mean_grad = torch.autograd.grad(reference.mean(), params, retain_graph=True)
    second_moment_grad = torch.autograd.grad(reference.square().mean(), params)
    endpoints = {id(p) for p in model.gamma_schedule.endpoint_parameters()}
    for p, got, mean, second in zip(params, actual, mean_grad, second_moment_grad):
        expected = mean if id(p) in endpoints else second
        torch.testing.assert_close(got, expected, rtol=1e-10, atol=1e-10)


def test_reconstruction_does_not_train_schedule_shape():
    model = make_model("algo.schedule_hidden_dim=16")
    gamma, prime, _ = model._training_gamma(torch.zeros(2), 2)
    (gamma.sum() + prime.sum()).backward()
    endpoints = {id(p) for p in model.gamma_schedule.endpoint_parameters()}
    for p in model.gamma_schedule.parameters():
        if id(p) not in endpoints:
            assert p.grad is None or torch.count_nonzero(p.grad) == 0


def test_adaptive_split_recovers_variance_from_single_example_updates():
    model = make_model("algo.split_ema=0.5")
    for i in range(30):
        model._update_split_statistics(torch.tensor([float(i % 2) * 4]), torch.tensor([1., 2.]))
    assert model.recon_std_ema > 1
    assert model._split_size(128) > 32


def test_schedule_optimizer_groups_cover_each_parameter_once():
    model = make_model()
    optimizers, _ = model.configure_optimizers()
    groups = optimizers[0].param_groups
    all_ids = [id(p) for group in groups for p in group["params"]]
    assert len(all_ids) == len(set(all_ids))
    assert set(all_ids) == {id(p) for p in model._get_parameters()}
    assert groups[1]["weight_decay"] == 0
    assert groups[1]["initial_lr"] == model.config.algo.special_lr
    assert {id(p) for p in groups[2]["params"]} == {id(p) for p in model.gamma_schedule.endpoint_parameters()}


def test_state_roundtrip_preserves_schedule_and_split_statistics(tmp_path):
    model = make_model()
    model._update_split_statistics(torch.tensor([1., 3.]), torch.tensor([2., 5.]))
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = make_model()
    restored.load_state_dict(torch.load(path, weights_only=True))
    t = torch.linspace(0, 1, 9)
    torch.testing.assert_close(model._gamma(t), restored._gamma(t))
    assert model._split_size(128) == restored._split_size(128)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mixed_precision_keeps_embedding_and_input_projections_float32():
    model = make_model().cuda()
    z = torch.randn(2, 4, model.config.algo.embedding_dim, device="cuda")
    gamma = torch.tensor([-3., 4.], device="cuda")
    outputs = []
    handle = model.backbone.noisy_projection.register_forward_hook(
        lambda module, args, result: outputs.append(result.dtype)
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        log_probs = model.denoise(z, gamma)
        embedding = model.backbone.embed_tokens(log_probs.exp())
    handle.remove()
    assert outputs == [torch.float32]
    assert embedding.dtype == log_probs.dtype == torch.float32
    assert torch.isfinite(log_probs).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA autocast")
def test_bootstrap_under_outer_autocast_updates_transformer_weights():
    model = make_model("algo.self_condition_rate=0.25").cuda().train()
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]] * 4, device="cuda")
    optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=1e-3)
    attention_before = model.backbone.blocks[0].attn_qkv.weight.detach().clone()
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        # Lightning holds this scope open across bootstrap and training passes.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model._loss(tokens, torch.ones_like(tokens), train_mode=True).loss
        loss.backward()
        gradient = model.backbone.output_layer.linear.weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
        optimizer.step()
    assert model.backbone.output_layer.linear.weight.norm() > 0
    assert not torch.equal(attention_before, model.backbone.blocks[0].attn_qkv.weight)


@pytest.mark.parametrize("rate,expected_nfe", [(0.25, 6), (0.0, 3)])
def test_sampling_bootstrap_matches_reported_denoiser_calls(monkeypatch, rate, expected_nfe):
    model = make_model(f"algo.self_condition_rate={rate}")
    model.eval()
    calls = []
    original = model.sample_replaid_denoise

    def capture(z, t, context=None):
        calls.append((t.clone(), context))
        return original(z, t, context)

    monkeypatch.setattr(model, "sample_replaid_denoise", capture)
    result = model.generate_samples(num_samples=2, num_steps=2, return_result=True)
    assert result.metadata["nfe"] == len(calls) == expected_nfe
    if rate:
        for first, second in zip(calls[::2], calls[1::2]):
            assert first[1] is None
            assert second[1] is not None
            assert torch.equal(first[0], second[0])

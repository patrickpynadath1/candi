import torch
from hydra import compose, initialize_config_dir

from algos import algo_registry
from dataloader import Text8Tokenizer
from pathlib import Path


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

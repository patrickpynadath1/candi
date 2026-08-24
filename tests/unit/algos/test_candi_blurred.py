from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

import main  # noqa: F401 -- config resolver registration
from algos import algo_registry
from dataloader import Text8Tokenizer


CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")


def make_model(*overrides):
    base = [
        "algo=candi_blurred",
        "model=tiny",
        "data=ptb",
        "training.ema=0",
        "sampling.predictor=candi_entropy_bounded",
        "model.length=6",
        "model.hidden_size=8",
        "model.cond_dim=8",
        "model.n_blocks=1",
        "model.n_heads=2",
        "model.dropout=0",
        *overrides,
    ]
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        config = compose(config_name="config", overrides=base)
    return algo_registry.get("candi_blurred")(config, Text8Tokenizer())


def test_candi_blurred_corruption_keeps_discrete_reveals_and_blurs_positions():
    model = make_model()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    valid = torch.tensor([[0, 0, 1, 1, 1, 1]])
    result = model.q_xt(tokens, torch.zeros(1, 1), valid_tokens=valid)
    clean = model.backbone.get_embedding(tokens)
    assert torch.equal(result["xt"][:, :2], clean[:, :2])
    assert result["xt"].shape == clean.shape
    assert result["rank_noise"].shape == (1,)
    assert torch.isfinite(result["xt"]).all()


def test_candi_blurred_sampling_prediction_is_argmax():
    model = make_model()
    probabilities = torch.full((2, 6, model.vocab_size - 1), 0.01)
    probabilities[..., 7] = 0.8
    assert torch.equal(
        model.sampling_clean_tokens(probabilities),
        torch.full((2, 6), 7),
    )


def test_candi_blurred_can_use_linearized_crossing_clock():
    model = make_model(
        "model.length=89",
        "algo.blurring.time_warp=linearized_crossings",
        "algo.blurring.crossing_max_decile_fraction=0.15",
    )
    grid = torch.linspace(0, 1, 10001, dtype=torch.float64)
    a, sigma, _, _ = model.mode_coefficients(grid, 89)
    crossings = grid[
        (a.square() / sigma[:, None].square().clamp_min(1e-30) <= 1)
        .to(torch.int8)
        .argmax(dim=0)
    ]
    histogram = torch.histc(crossings, bins=10, min=0, max=1)
    assert int(histogram.max()) <= 10
    assert torch.allclose(crossings[0], crossings.new_tensor(0.95), atol=2e-4)
    assert torch.allclose(crossings[-1], crossings.new_tensor(0.05), atol=2e-4)


def test_candi_blurred_continuous_transition_is_finite_and_clamps_clean_sites():
    model = make_model()
    state = model.prior_sample(2, 6)
    clean_mask = torch.tensor(
        [[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool
    )
    prompt = model.backbone.get_embedding(
        torch.tensor([[1, 2], [2, 1]]), detach=True
    )
    state[:, :2] = prompt
    continued, probabilities = model.sample_candi_continuous_transition(
        state,
        torch.tensor(0.7),
        time_s=torch.tensor(0.4),
        sigma_s=torch.tensor(0.7),
        sigma_t=torch.tensor(0.4),
        clean_mask=clean_mask,
    )
    assert continued.shape == state.shape
    assert probabilities.shape == (2, 6, model.vocab_size - 1)
    assert torch.equal(continued[clean_mask], state[clean_mask])
    assert torch.isfinite(continued).all()
    assert torch.allclose(probabilities.sum(-1), torch.ones(2, 6), atol=1e-5)


def test_candi_blurred_zero_noise_scale_is_deterministic():
    model = make_model("algo.continuous_noise_scale=0")
    state = model.prior_sample(2, 6)
    clean_mask = torch.zeros(2, 6, dtype=torch.bool)

    torch.manual_seed(1)
    first, _ = model.sample_candi_continuous_transition(
        state.clone(), torch.tensor(0.7), time_s=torch.tensor(0.4),
        sigma_s=torch.tensor(0.7), sigma_t=torch.tensor(0.4),
        clean_mask=clean_mask,
    )
    torch.manual_seed(2)
    second, _ = model.sample_candi_continuous_transition(
        state.clone(), torch.tensor(0.7), time_s=torch.tensor(0.4),
        sigma_s=torch.tensor(0.7), sigma_t=torch.tensor(0.4),
        clean_mask=clean_mask,
    )

    assert torch.equal(first, second)


def test_candi_blurred_ddim_transition_is_deterministic():
    model = make_model("algo.continuous_transition=ddim")
    state = model.prior_sample(2, 6)
    clean_mask = torch.zeros(2, 6, dtype=torch.bool)

    torch.manual_seed(1)
    first, _ = model.sample_candi_continuous_transition(
        state.clone(), torch.tensor(0.7), time_s=torch.tensor(0.4),
        sigma_s=torch.tensor(0.7), sigma_t=torch.tensor(0.4),
        clean_mask=clean_mask,
    )
    torch.manual_seed(2)
    second, _ = model.sample_candi_continuous_transition(
        state.clone(), torch.tensor(0.7), time_s=torch.tensor(0.4),
        sigma_s=torch.tensor(0.7), sigma_t=torch.tensor(0.4),
        clean_mask=clean_mask,
    )

    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def test_candi_blurred_transition_can_return_untempered_probabilities():
    model = make_model("algo.temp=0.1", "algo.continuous_transition=ddim")
    state = model.prior_sample(2, 6)
    _, tempered, untempered = model.sample_candi_continuous_transition(
        state, torch.tensor(0.7), time_s=torch.tensor(0.4),
        sigma_s=torch.tensor(0.7), sigma_t=torch.tensor(0.4),
        clean_mask=torch.zeros(2, 6, dtype=torch.bool),
        return_untempered=True,
    )

    assert tempered.shape == untempered.shape
    assert torch.allclose(untempered.sum(-1), torch.ones(2, 6), atol=1e-5)
    assert not torch.allclose(tempered, untempered)


def test_candi_blurred_loss_backpropagates_through_real_backbone():
    model = make_model()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    result = model._loss(tokens, torch.ones_like(tokens))
    result.loss.backward()
    gradients = [p.grad for p in model.backbone.parameters() if p.grad is not None]
    assert torch.isfinite(result.loss)
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

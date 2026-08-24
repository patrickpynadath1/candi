from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

import main  # noqa: F401 -- config resolver registration
from algos import algo_registry
from algos.blurring_diffusion import dct_matrix
from dataloader import Text8Tokenizer


CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")


def make_model(*overrides):
    base = [
        "algo=blurring_diffusion",
        "model=tiny",
        "data=ptb",
        "training.ema=0",
        "sampling.predictor=blurring_ode",
        "model.length=6",
        "model.hidden_size=8",
        "model.cond_dim=8",
        "model.n_blocks=1",
        "model.n_heads=2",
        "model.dropout=0",
        "algo.input_embedding_dim=4",
        "algo.output_embedding_dim=8",
        *overrides,
    ]
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        config = compose(config_name="config", overrides=base)
    return algo_registry.get("blurring_diffusion")(config, Text8Tokenizer())


def test_dct_is_orthonormal_and_round_trips():
    basis = dct_matrix(17, dtype=torch.float64)
    identity = torch.eye(17, dtype=torch.float64)
    assert torch.allclose(basis.t() @ basis, identity, atol=1e-12)
    values = torch.randn(17, 5, dtype=torch.float64)
    assert torch.allclose(basis @ (basis.t() @ values), values, atol=1e-12)


def test_high_frequency_modes_die_first_and_prior_is_clean():
    model = make_model()
    t = torch.tensor([0.0, 0.5, 1.0])
    a, sigma, _, _ = model.mode_coefficients(t, 9)
    assert torch.allclose(a[0], torch.ones_like(a[0]))
    assert a[1, -1] < a[1, 1] < a[1, 0]
    assert torch.allclose(a[-1, 0], torch.tensor(0.03), atol=1e-5)
    assert sigma[0] < 1e-5
    assert sigma[-1] > 0.999


def test_linearized_schedule_spreads_snr_crossings_uniformly():
    model = make_model("model.length=89")
    grid = torch.linspace(0, 1, 10001, dtype=torch.float64)
    a, sigma, _, _ = model.mode_coefficients(grid, 89)
    snr = a.square() / sigma[:, None].square().clamp_min(1e-30)
    crossings = grid[(snr <= 1).to(torch.int8).argmax(dim=0)]
    expected = torch.linspace(0.95, 0.05, 89, dtype=torch.float64)
    assert torch.allclose(crossings, expected, atol=3e-4)
    histogram = torch.histc(crossings, bins=10, min=0, max=1)
    assert int(histogram.max()) <= 10


def test_linearized_schedule_preserves_endpoints_and_markov_variance():
    model = make_model("model.length=89")
    grid = torch.linspace(0, 1, 1001, dtype=torch.float64)
    a, sigma, _, _ = model.mode_coefficients(grid, 89)
    assert torch.equal(a[0], torch.ones_like(a[0]))
    assert sigma[0] == 0
    assert torch.allclose(sigma[-1], torch.tensor(1.0, dtype=sigma.dtype))
    ratio = a[1:] / a[:-1].clamp_min(torch.finfo(a.dtype).tiny)
    d2 = sigma[1:, None].square() - ratio.square() * sigma[:-1, None].square()
    assert d2.min() >= -1e-10


def test_original_geometric_clock_remains_selectable():
    model = make_model(
        "algo.time_warp=none",
        "algo.crossing_max_decile_fraction=0.45",
        "model.length=89",
    )
    grid = torch.linspace(0, 1, 10001, dtype=torch.float64)
    a, sigma, _, _ = model.mode_coefficients(grid, 89)
    crossings = grid[
        (a.square() / sigma[:, None].square().clamp_min(1e-30) <= 1)
        .to(torch.int8)
        .argmax(dim=0)
    ]
    histogram = torch.histc(crossings, bins=10, min=0, max=1)
    assert int(histogram.max()) == 37


def test_reverse_posterior_reblurs_oracle_prediction_to_target_time():
    model = make_model()
    length, width = 6, 3
    clean = torch.randn(2, length, width)
    t = torch.tensor([0.8, 0.6])
    s = torch.tensor([0.35, 0.2])
    a_t, _, _, _ = model.separable_coefficients(t, length, width)
    a_s, _, _, _ = model.separable_coefficients(s, length, width)
    expected_state = a_t * clean
    mean, _ = model.reverse_mode_moments(expected_state, clean, t, s, length)
    assert torch.allclose(mean, a_s * clean, atol=2e-5)


def test_omitting_reblur_reproduces_wrong_oracle_landing():
    model = make_model("+sampling.omit_reblur=true")
    length = 6
    clean = torch.randn(1, length, 2)
    t, s = torch.tensor([0.8]), torch.tensor([0.35])
    a_t, _, _, _ = model.separable_coefficients(t, length, clean.shape[-1])
    a_s, _, _, _ = model.separable_coefficients(s, length, clean.shape[-1])
    mean, _ = model.reverse_mode_moments(a_t * clean, clean, t, s, length)
    assert not torch.allclose(mean, a_s * clean, atol=1e-4)


def test_reverse_posterior_identity_at_clean_endpoint():
    model = make_model()
    length = 6
    state = torch.randn(1, length, 2)
    clean = torch.randn_like(state)
    mean, variance = model.reverse_mode_moments(
        state, clean, torch.tensor([0.7]), torch.tensor([0.0]), length
    )
    assert torch.allclose(mean, clean, atol=1e-6)
    assert torch.equal(variance, torch.zeros_like(variance))


def test_reverse_posterior_matches_forward_marginal_variance():
    model = make_model()
    length = 6
    t, s = torch.tensor([0.75]), torch.tensor([0.3])
    clean = torch.randn(1, length, 1)
    a_t, sigma_t, _, _ = model.separable_coefficients(t, length, clean.shape[-1])
    _, sigma_s, _, _ = model.separable_coefficients(s, length, clean.shape[-1])
    # Recover the linear state coefficient by comparing two states one unit apart.
    base = a_t * clean
    mean_0, posterior_variance = model.reverse_mode_moments(base, clean, t, s, length)
    mean_1, _ = model.reverse_mode_moments(
        base + torch.ones_like(base), clean, t, s, length
    )
    state_coefficient = mean_1 - mean_0
    marginal_variance = (
        state_coefficient.square() * sigma_t[:, None, None].square()
        + posterior_variance
    )
    assert torch.allclose(
        marginal_variance,
        sigma_s[:, None, None].square().expand_as(marginal_variance),
        atol=1e-6,
    )


def test_corruption_preserves_conditioning_and_trains_input_table():
    model = make_model()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    valid = torch.tensor([[0, 0, 1, 1, 1, 1]])
    clean = model.backbone.get_embedding(tokens).detach()
    noisy = model.q_xt(tokens, torch.tensor([0.7]), valid)
    assert torch.equal(noisy[:, :2], clean[:, :2])
    assert not torch.equal(noisy[:, 2:], clean[:, 2:])

    loss = model._loss(tokens, valid).loss
    loss.backward()
    gradient = model.backbone.input_embedding.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0
    norms = model.backbone.normalized_input_table().norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_detached_forward_table_has_no_gradient_from_corruption():
    model = make_model("algo.detach_forward_embedding=true")
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    valid = torch.ones_like(tokens)
    model._loss(tokens, valid).loss.backward()
    assert model.backbone.input_embedding.weight.grad is None


def test_conditional_sampler_preserves_prompt():
    model = make_model()
    prompt = torch.tensor([[1, 2, 0, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    samples = model.generate_samples(
        1, num_steps=2, prompt_tokens=prompt, prompt_mask=mask
    )
    assert samples.shape == prompt.shape
    assert torch.equal(samples[mask], prompt[mask])


def test_greedy_sampling_embeds_argmax_instead_of_probability_mean():
    model = make_model("sampling.greedy_clean_estimate=true")
    probabilities = torch.full((1, 1, model.vocab_size), 0.05)
    probabilities[..., 3] = 0.40
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    estimate = model.sampling_clean_estimate(probabilities)
    expected = model.backbone.get_embedding(probabilities.argmax(-1), detach=True)
    blended = model.backbone.get_embedding(probabilities, detach=True)
    assert torch.equal(estimate, expected)
    assert not torch.allclose(estimate, blended)


def test_feature_columns_die_in_declared_order_and_rho_endpoints_work():
    model = make_model()
    t = torch.tensor([0.6])
    joint, _, _, _ = model.separable_coefficients(t, 6, 4)
    assert torch.all(joint[0, :, 0] > joint[0, :, -1])

    positional = make_model("algo.feature_rho=1.0")
    joint, sigma, _, _ = positional.separable_coefficients(t, 6, 4)
    a_pos, expected_sigma, _, _ = positional.mode_coefficients(t, 6)
    assert torch.allclose(joint, a_pos[:, :, None].expand_as(joint))
    assert torch.equal(sigma, expected_sigma)

    feature_only = make_model("algo.feature_rho=0.0")
    joint, _, _, _ = feature_only.separable_coefficients(t, 6, 4)
    assert torch.allclose(joint[:, :1], joint)


def test_feature_deaths_span_the_requested_early_to_late_window():
    model = make_model("algo.input_embedding_dim=16")
    grid = torch.linspace(0, 1, 10001, dtype=torch.float64)
    joint, sigma, _, _ = model.separable_coefficients(grid, 6, 16)
    # k=0 includes positional attenuation as it will in the real process.
    crossings = grid[
        (joint[:, 0].square() / sigma[:, None].square().clamp_min(1e-30) <= 1)
        .to(torch.int8)
        .argmax(0)
    ]
    assert 0.85 <= crossings[0] <= 0.95
    assert 0.05 <= crossings[-1] <= 0.15
    assert torch.all(crossings[:-1] >= crossings[1:])


def test_input_table_is_centered_normalized_and_column_aware():
    model = make_model()
    table = model.backbone.data_input_table()
    assert torch.allclose(table.mean(0), torch.zeros(4), atol=2e-4)
    assert torch.allclose(table.norm(dim=-1), torch.ones(len(table)), atol=1e-6)
    norms = table.norm(dim=0)
    assert norms[0] > norms[-1]


def test_padding_values_cannot_leak_into_generated_span():
    model = make_model()
    valid = torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.bool)
    tokens_a = torch.tensor([[1, 2, 3, 4, 5, 6]])
    tokens_b = torch.tensor([[7, 8, 3, 4, 5, 6]])
    torch.manual_seed(11)
    noisy_a = model.q_xt(tokens_a, torch.tensor([0.5]), valid)
    torch.manual_seed(11)
    noisy_b = model.q_xt(tokens_b, torch.tensor([0.5]), valid)
    assert torch.equal(noisy_a[:, 2:], noisy_b[:, 2:])

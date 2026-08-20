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


def test_reverse_posterior_reblurs_oracle_prediction_to_target_time():
    model = make_model()
    length, width = 6, 3
    clean = torch.randn(2, length, width)
    t = torch.tensor([0.8, 0.6])
    s = torch.tensor([0.35, 0.2])
    a_t, _, _, _ = model.mode_coefficients(t, length)
    a_s, _, _, _ = model.mode_coefficients(s, length)
    expected_state = a_t[:, :, None] * clean
    mean, _ = model.reverse_mode_moments(
        expected_state, clean, t, s, length
    )
    assert torch.allclose(mean, a_s[:, :, None] * clean, atol=2e-5)


def test_omitting_reblur_reproduces_wrong_oracle_landing():
    model = make_model("+sampling.omit_reblur=true")
    length = 6
    clean = torch.randn(1, length, 2)
    t, s = torch.tensor([0.8]), torch.tensor([0.35])
    a_t, _, _, _ = model.mode_coefficients(t, length)
    a_s, _, _, _ = model.mode_coefficients(s, length)
    mean, _ = model.reverse_mode_moments(
        a_t[:, :, None] * clean, clean, t, s, length
    )
    assert not torch.allclose(mean, a_s[:, :, None] * clean, atol=1e-4)


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
    a_t, sigma_t, _, _ = model.mode_coefficients(t, length)
    _, sigma_s, _, _ = model.mode_coefficients(s, length)
    clean = torch.randn(1, length, 1)
    # Recover the linear state coefficient by comparing two states one unit apart.
    base = a_t[:, :, None] * clean
    mean_0, posterior_variance = model.reverse_mode_moments(
        base, clean, t, s, length
    )
    mean_1, _ = model.reverse_mode_moments(
        base + torch.ones_like(base), clean, t, s, length
    )
    state_coefficient = (mean_1 - mean_0).squeeze(-1)
    marginal_variance = (
        state_coefficient.square() * sigma_t[:, None].square()
        + posterior_variance
    )
    assert torch.allclose(
        marginal_variance, sigma_s[:, None].square().expand_as(marginal_variance),
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
    model = make_model("+sampling.greedy_clean_estimate=true")
    probabilities = torch.full((1, 1, model.vocab_size), 0.05)
    probabilities[..., 3] = 0.40
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    estimate = model.sampling_clean_estimate(probabilities)
    expected = model.backbone.get_embedding(probabilities.argmax(-1), detach=True)
    blended = model.backbone.get_embedding(probabilities, detach=True)
    assert torch.equal(estimate, expected)
    assert not torch.allclose(estimate, blended)

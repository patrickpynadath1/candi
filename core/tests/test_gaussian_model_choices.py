"""Continuous-network input preconditioning and time-conditioning contracts."""

import pytest
import torch

from core.models.gaussian_conditioning import TIME_CONDITIONING
from core.tests import networks
from core.tests.networks import path_call


def make_model(**kwargs):
    return networks.continuous(7, state_dim=5, **kwargs)


def test_default_preconditioning_and_time_match_original_path():
    model = make_model()
    state = torch.randn(2, 3, 5)
    time = torch.tensor([0.2, 1.0])
    alpha, sigma = torch.tensor([1.0, 0.5]), torch.tensor([2.0, 0.8])
    observed = {}
    handles = [
        model.input.projection.register_forward_pre_hook(
            lambda module, args: observed.update(projected=args[0].detach())
        ),
        model.time.embedder.register_forward_pre_hook(
            lambda module, args: observed.update(condition=args[0].detach())
        ),
    ]
    path_call(model, state, time, sigma, alpha)
    for handle in handles:
        handle.remove()
    expected = state @ torch.nn.functional.normalize(model.input.embedding, dim=-1)
    expected *= torch.rsqrt(alpha.square() + sigma.square())[:, None, None]
    torch.testing.assert_close(observed["projected"], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        observed["condition"],
        -torch.log1p(-time.clamp(0, 1 - model.time.epsilon)),
        rtol=0,
        atol=0,
    )


def test_no_preconditioning_retains_projection_and_trainable_embedding():
    model = make_model(preconditioning="none", coordinate="time")
    state = torch.randn(2, 3, 5)
    time = torch.tensor([0.2, 0.8])
    one = torch.ones(2)
    original = path_call(model, state, time, one, one)
    rescaled_coefficients = path_call(model, state, time, 3 * one, 5 * one)
    torch.testing.assert_close(original, rescaled_coefficients, rtol=0, atol=0)
    original.square().mean().backward()
    for parameter in (model.input.embedding, model.input.projection.weight):
        assert parameter.grad is not None
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("conditioning", ["log_effective_noise", "log_snr"])
def test_effective_noise_conditioning_is_invariant_to_path_rescaling(conditioning):
    model = make_model(coordinate=conditioning)
    state = torch.randn(2, 3, 5)
    time = torch.tensor([0.2, 0.8])
    sigma, alpha = torch.tensor([0.4, 2.0]), torch.tensor([0.8, 0.3])
    scale = torch.tensor([0.5, 2.0])
    torch.testing.assert_close(
        path_call(model, state, time, sigma, alpha),
        path_call(model, state * scale[:, None, None], time, sigma * scale, alpha * scale),
    )


@pytest.mark.parametrize("name", TIME_CONDITIONING)
def test_conditioning_endpoints_are_finite_and_checkpoint_layout_unchanged(name):
    model = make_model(coordinate=name)
    model.load_state_dict(make_model().state_dict(), strict=True)
    result = path_call(
        model,
        torch.randn(2, 3, 5),
        torch.tensor([0.0, 1.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 0.0]),
    )
    assert torch.isfinite(result).all()


def test_log_conditioning_coordinates():
    time, sigma, alpha = torch.tensor([0.4]), torch.tensor([2.0]), torch.tensor([0.5])
    assert TIME_CONDITIONING["log_sigma"](
        time, sigma, alpha, 1e-6
    ).item() == pytest.approx(torch.log(sigma).item())
    assert TIME_CONDITIONING["log_effective_noise"](
        time, sigma, alpha, 1e-6
    ).item() == pytest.approx(torch.log(sigma / alpha).item())
    assert TIME_CONDITIONING["log_snr"](
        time, sigma, alpha, 1e-6
    ).item() == pytest.approx(torch.log(alpha.square() / sigma.square()).item())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"preconditioning": "bad"},
        {"coordinate": "bad"},
        {"epsilon": 0},
    ],
)
def test_invalid_choices_rejected(kwargs):
    with pytest.raises(ValueError):
        make_model(**kwargs)

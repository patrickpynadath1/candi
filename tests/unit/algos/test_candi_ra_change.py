import pytest
import torch

from algos.candi_ra_change import CANDIRAChange
from algos.neighbor_blurring import blend_positions, gaussian_position_kernel


def _bare(**attrs):
    model = object.__new__(CANDIRAChange)
    torch.nn.Module.__init__(model)
    for name, value in attrs.items():
        setattr(model, name, value)
    return model


def test_gaussian_position_kernel_is_row_stochastic_and_symmetric_inside():
    kernel = gaussian_position_kernel(7, 1.5)
    assert torch.allclose(kernel.sum(-1), torch.ones(7))
    assert torch.all(kernel >= 0)
    assert torch.allclose(kernel[3], kernel[3].flip(0))


def test_zero_bandwidth_is_exact_identity_limit():
    values = torch.randn(2, 5, 3)
    kernel = gaussian_position_kernel(5, 0)
    assert torch.equal(kernel, torch.eye(5))
    assert torch.equal(blend_positions(values, kernel), values)


def test_blending_spreads_a_position_to_neighbors():
    values = torch.zeros(1, 5, 1)
    values[0, 2, 0] = 1
    blurred = blend_positions(values, gaussian_position_kernel(5, 1.0))
    assert blurred[0, 2, 0] > blurred[0, 1, 0] > blurred[0, 0, 0]
    assert blurred[0, 3, 0] == pytest.approx(blurred[0, 1, 0])


def test_q_xt_uses_blended_mean_only_for_unrevealed_positions(monkeypatch):
    model = _bare(
        vocab_size=4,
        spatial_kernel=gaussian_position_kernel(3, 1.0),
    )
    model.discrete_noising = lambda x, _alpha: torch.tensor([[0, 2, 2]])
    model.get_continuous_from_discrete_noise = lambda noise: torch.zeros_like(noise)
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.zeros_like(value))
    tokens = torch.tensor([[0, 1, 2]])

    result = model.q_xt(tokens, torch.tensor([[0.5]]))
    onehot = torch.nn.functional.one_hot(tokens, 3).float()
    expected_blur = blend_positions(onehot, model.spatial_kernel)
    assert torch.equal(result["xt"][:, 0], onehot[:, 0])
    assert torch.allclose(result["xt"][:, 1], expected_blur[:, 1])
    assert torch.equal(result["xt"][:, 2], onehot[:, 2])


def test_spatial_kernel_is_a_registered_buffer():
    model = _bare()
    model.register_buffer("spatial_kernel", gaussian_position_kernel(3, 0.5))
    assert "spatial_kernel" in dict(model.named_buffers())

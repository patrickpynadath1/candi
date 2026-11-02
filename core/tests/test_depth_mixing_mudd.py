"""Behavioral checks for independent, positionwise MUDD depth routing."""

import pytest
import torch

from core.models.depth_mixing.mudd import MUDDDepthMixer


def test_initialization_preserves_latest_representation_and_gradient():
    history = torch.randn(2, 5, 3, 8, requires_grad=True)
    mixer = MUDDDepthMixer(8, 3)
    out = mixer(history)
    torch.testing.assert_close(out, history[:, :, -1, None].expand(-1, -1, 4, -1))
    out.sum().backward()
    torch.testing.assert_close(
        history.grad[:, :, :-1], torch.zeros_like(history.grad[:, :, :-1])
    )
    torch.testing.assert_close(
        history.grad[:, :, -1], torch.full_like(history.grad[:, :, -1], 4)
    )
    assert mixer.down.weight.grad.abs().sum() > 0
    assert mixer.static_weights.grad.abs().sum() > 0


def test_streams_can_select_distinct_depths_without_normalization():
    mixer = MUDDDepthMixer(8, 3)
    with torch.no_grad():
        mixer.static_weights.copy_(
            torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 2.0]]
            )
        )
    history = torch.randn(2, 5, 3, 8)
    expected = torch.stack(
        (
            history[:, :, 0],
            history[:, :, 1],
            history[:, :, 2],
            -history[:, :, 0] + 2 * history[:, :, 2],
        ),
        dim=2,
    )
    torch.testing.assert_close(mixer(history), expected)


def test_dynamic_router_is_positionwise_and_trains_both_projections():
    torch.manual_seed(42)
    mixer = MUDDDepthMixer(8, 3)
    with torch.no_grad():
        mixer.down.weight.normal_(std=0.1)
    history = torch.randn(2, 5, 3, 8, requires_grad=True)
    original = mixer(history)
    changed = history.detach().clone()
    changed[:, 0] += 5
    torch.testing.assert_close(mixer(changed)[:, 1:], original[:, 1:])
    assert not torch.allclose(mixer(changed)[:, 0], original[:, 0])
    original.square().sum().backward()
    for parameter in mixer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert history.grad[:, :, :-1].abs().sum() > 0


def test_bfloat16_autocast_keeps_history_dtype_and_finite_values():
    mixer = MUDDDepthMixer(8, 3)
    history = torch.randn(2, 5, 3, 8, dtype=torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = mixer(history)
    assert output.dtype == history.dtype
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("shape", [(2, 5, 8), (2, 5, 2, 8), (2, 5, 3, 9)])
def test_incompatible_history_rejected(shape):
    with pytest.raises(ValueError, match="history must have shape"):
        MUDDDepthMixer(8, 3)(torch.randn(shape))

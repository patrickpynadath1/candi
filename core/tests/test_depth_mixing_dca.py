import pytest
import torch

from core.models.depth_mixing.dca import DCA


def test_paper_initialization_recovers_residual_sum():
    history = torch.randn(2, 5, 3, 4)
    mixer = DCA(dim=4, num_sources=3)
    expected = history.sum(dim=-2).unsqueeze(-2).expand(-1, -1, 4, -1)
    torch.testing.assert_close(mixer(history), expected)


def test_independent_feature_and_source_conditioned_weights():
    history = torch.tensor([[[[1.0, 2.0], [-2.0, 3.0]], [[-1.0, 2.0], [2.0, -3.0]]]])
    mixer = DCA(dim=2, num_sources=2)
    with torch.no_grad():
        mixer.static_weights.copy_(torch.arange(12).reshape(3, 2, 2) / 10)
        mixer.gate_vectors.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5]]))
    expected = torch.empty(1, 2, 4, 2)
    for position in range(2):
        for stream in range(3):
            expected[0, position, stream] = sum(
                history[0, position, source]
                * (
                    mixer.static_weights[stream, source]
                    + torch.relu(
                        mixer.gate_vectors[stream] @ history[0, position, source]
                    )
                )
                for source in range(2)
            )
    expected[:, :, 3] = expected[:, :, 0]
    torch.testing.assert_close(mixer(history), expected)


@pytest.mark.parametrize("gate_init_std, active", [(0.0, False), (1e-3, True)])
def test_dynamic_gate_initialization_gradient(gate_init_std, active):
    torch.manual_seed(42)
    mixer = DCA(dim=8, num_sources=4, gate_init_std=gate_init_std)
    history = torch.randn(2, 7, 4, 8, requires_grad=True)
    mixer(history).square().mean().backward()
    assert bool(mixer.gate_vectors.grad.abs().sum() > 0) is active
    assert (history.grad.abs().sum(dim=-1) > 0).all()
    assert mixer.static_weights.grad.abs().sum() > 0


def test_positions_do_not_exchange_information():
    torch.manual_seed(2)
    mixer = DCA(dim=4, num_sources=2, gate_init_std=0.1)
    history = torch.randn(1, 3, 2, 4)
    changed = history.clone()
    changed[:, 1] += 5
    torch.testing.assert_close(mixer(history)[:, [0, 2]], mixer(changed)[:, [0, 2]])


@pytest.mark.parametrize("shape", [(1, 2, 4), (1, 2, 2, 4), (1, 2, 3, 5)])
def test_rejects_wrong_history_shape(shape):
    with pytest.raises(ValueError, match="DCA expects history"):
        DCA(dim=4, num_sources=3)(torch.zeros(shape))


def test_residual_contribution_wiring_matches_standard_block_at_initialization():
    torch.manual_seed(1)
    x = torch.randn(2, 3, 4)
    ordinary = x
    history = [x]
    for depth in range(3):
        mixer = DCA(dim=4, num_sources=depth + 1)
        q, k, v, residual = mixer(torch.stack(history, dim=-2)).unbind(-2)
        # Stand-ins keep this test about residual bookkeeping, not attention.
        attention = (q + k + v).sin()
        ff = (residual + attention).tanh()
        history.append(attention + ff)
        ordinary_attention = (3 * ordinary).sin()
        ordinary = (
            ordinary + ordinary_attention + (ordinary + ordinary_attention).tanh()
        )
    torch.testing.assert_close(torch.stack(history, dim=-2).sum(dim=-2), ordinary)

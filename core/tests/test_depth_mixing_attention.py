import pytest
import torch

from core.models.depth_mixing.attention import DepthAttention


def test_zero_gate_preserves_latest_state_and_can_learn():
    torch.manual_seed(10)
    mixer = DepthAttention(12, 3, key_dim=4)
    history = torch.randn(2, 5, 3, 12, requires_grad=True)
    output = mixer(history)
    torch.testing.assert_close(
        output, history[..., -1, :].unsqueeze(-2).expand(-1, -1, 4, -1)
    )
    output.square().sum().backward()
    assert torch.count_nonzero(mixer.gate.grad) == 4
    assert torch.count_nonzero(mixer.query.weight.grad) == 0
    assert torch.count_nonzero(mixer.key.weight.grad) == 0


def test_depth_attention_has_no_cross_position_or_batch_dependency():
    torch.manual_seed(11)
    mixer = DepthAttention(12, 3, key_dim=4, gate_init=1.0)
    history = torch.randn(2, 5, 3, 12, requires_grad=True)
    output = mixer(history)
    output[0, 2].square().sum().backward()
    assert torch.count_nonzero(history.grad[1]) == 0
    assert torch.count_nonzero(history.grad[0, [0, 1, 3, 4]]) == 0
    assert torch.count_nonzero(history.grad[0, 2]) == 3 * 12
    assert torch.count_nonzero(mixer.query.weight.grad) > 0
    assert torch.count_nonzero(mixer.key.weight.grad) > 0
    assert not torch.allclose(output[..., 0, :], output[..., 1, :])


def test_identical_source_states_are_preserved_even_with_open_gates():
    mixer = DepthAttention(12, 4, key_dim=4, gate_init=1.0)
    state = torch.randn(2, 5, 12)
    output = mixer(state.unsqueeze(-2).expand(-1, -1, 4, -1))
    torch.testing.assert_close(output, state.unsqueeze(-2).expand(-1, -1, 4, -1))


def test_depth_attention_reads_source_content_not_only_latest_state():
    torch.manual_seed(12)
    mixer = DepthAttention(12, 3, key_dim=4, gate_init=1.0)
    history = torch.randn(2, 5, 3, 12)
    changed = history.clone()
    changed[..., 0, :] += torch.randn(12)
    assert not torch.allclose(mixer(history), mixer(changed))


@pytest.mark.parametrize("shape", [(2, 5, 12), (2, 5, 2, 12), (2, 5, 3, 11)])
def test_depth_attention_rejects_wrong_history_shape(shape):
    with pytest.raises(ValueError, match="history must have shape"):
        DepthAttention(12, 3)(torch.randn(shape))

from types import SimpleNamespace

import pytest
import torch

from models.transposed_dit_cont import TransposedContDIT


def _config(position_encoding="none"):
    return SimpleNamespace(
        algo=SimpleNamespace(reserve_mask_embedding=False),
        model=SimpleNamespace(
            length=5,
            hidden_size=8,
            transpose_hidden_size=12,
            cond_dim=8,
            n_heads=2,
            n_blocks=1,
            dropout=0.0,
            coordinate_position_encoding=position_encoding,
        ),
    )


@pytest.mark.parametrize("position_encoding", ["none", "rope", "learned"])
def test_transposed_cont_dit_preserves_external_shapes(position_encoding):
    model = TransposedContDIT(_config(position_encoding), vocab_size=6)
    state = torch.randn(2, 5, 8)
    logits, reconstructed = model(
        state,
        discrete_noise=torch.tensor([0.2, 0.8]),
        continuous_noise=torch.tensor([0.3, 0.7]),
        reveal_mask=None,
        is_embed=True,
        ret_embedding=True,
    )
    assert logits.shape == (2, 5, 6)
    assert reconstructed.shape == state.shape
    assert torch.isfinite(logits).all()


def test_transposed_cont_dit_adapters_receive_gradients():
    model = TransposedContDIT(_config(), vocab_size=6)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    state = torch.randn(2, 5, 8)
    targets = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])

    # Successive updates open the zero-initialized output and AdaLN gates.
    for _ in range(3):
        optimizer.zero_grad()
        logits = model(
            state,
            discrete_noise=torch.tensor([0.2, 0.8]),
            continuous_noise=torch.tensor([0.3, 0.7]),
            reveal_mask=None,
            is_embed=True,
        )
        torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), targets.flatten()
        ).backward()
        optimizer.step()

    for parameter in (model.sequence_up.weight, model.sequence_down.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    attention_grad = model.blocks[0].attn_qkv.weight.grad
    assert attention_grad is not None
    assert torch.isfinite(attention_grad).all()
    assert torch.count_nonzero(attention_grad) > 0


def test_transposed_cont_dit_accepts_discrete_tokens():
    model = TransposedContDIT(_config(), vocab_size=6)
    logits = model(
        torch.tensor([[0, 1, 2, 3, 4]]),
        discrete_noise=torch.tensor([0.5]),
        continuous_noise=torch.tensor([0.5]),
        reveal_mask=None,
    )
    assert logits.shape == (1, 5, 6)


def test_transposed_cont_dit_rejects_the_wrong_sequence_length():
    model = TransposedContDIT(_config(), vocab_size=6)
    with pytest.raises(ValueError, match="requires sequence length 5"):
        model(
            torch.randn(1, 4, 8),
            discrete_noise=torch.tensor([0.5]),
            continuous_noise=torch.tensor([0.5]),
            reveal_mask=None,
            is_embed=True,
        )

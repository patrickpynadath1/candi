from types import SimpleNamespace

import torch

from algos.mdlm_perm import (
    MDLMPerm,
    permute_autoregressive_batch,
    sample_generation_order,
)


def test_permuted_batch_tracks_tokens_validity_and_decoupled_positions():
    tokens = torch.tensor([[9, 10, 11, 12, 13]])
    valid = torch.tensor([[1, 1, 1, 0, 1]])
    order = torch.tensor([[2, 0, 3, 1]])
    inputs, targets, output_valid, key_pos, query_pos = permute_autoregressive_batch(
        tokens, valid, order
    )
    assert inputs.tolist() == [[9, 12, 10, 13]]
    assert targets.tolist() == [[12, 10, 13, 11]]
    assert output_valid.tolist() == [[0, 1, 1, 1]]
    assert key_pos.tolist() == [[0, 3, 1, 4]]
    assert query_pos.tolist() == [[3, 1, 4, 2]]


def test_generation_order_modes_return_valid_permutations():
    for mode, block_size in (("random", -1), ("sequential", -1), ("block", 2)):
        order = sample_generation_order(
            3, 7, device=torch.device("cpu"), mode=mode, block_size=block_size
        )
        expected = torch.arange(7).expand_as(order)
        assert torch.equal(order.sort(-1).values, expected)


def test_permuted_algorithm_loss_uses_reordered_targets():
    model = object.__new__(MDLMPerm)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        algo={"order": "sequential", "ar_probability": 0.0, "block_size": -1}
    )
    model.mask_index = 4
    model.neg_infinity = -1e6

    def logits(tokens, _key_positions, _query_positions):
        result = torch.zeros(*tokens.shape, 5)
        result[..., 2] = 2
        result[..., 4] = -1e6
        return result

    model.permuted_logits = logits
    tokens = torch.tensor([[1, 2, 2, 2]])
    valid = torch.ones_like(tokens)
    loss = model._loss(tokens, valid)
    assert loss.loss.ndim == 0
    assert loss.num_tokens.item() == 3
    assert torch.isfinite(loss.loss)

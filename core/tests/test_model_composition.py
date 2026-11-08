"""Behavioral contracts for ModularDiT's interchangeable slots."""

import copy

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from core.models.modular import ModularDiT, SimplexInput
from core.tests import networks


@pytest.mark.parametrize("kind", ["token", "continuous", "sphere"])
def test_shared_backbone_matches_direct_block_and_head_execution(kind):
    """forward_hidden equals running the blocks and the head by hand."""
    torch.manual_seed(81)
    if kind == "token":
        model = networks.token(7, 16, 128, 2, 2, adaLN=True)
    elif kind == "continuous":
        model = networks.continuous(7, hidden_size=16, cond_dim=128, n_blocks=2, n_heads=2)
    else:
        model = networks.sphere(7, 7, 16, 128, 2, 2, temperature=True)
    # Zero-initialized AdaLN/head weights otherwise conceal backbone errors.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    reference = copy.deepcopy(model)
    hidden = torch.randn(2, 5, 16, requires_grad=True)
    condition = torch.randn(2, 128, requires_grad=True)
    ref_hidden = hidden.detach().clone().requires_grad_()
    ref_condition = condition.detach().clone().requires_grad_()
    positions = torch.tensor([4, 1, 0, 3, 2]).expand(2, -1)
    attention = torch.ones(5, 5, dtype=torch.bool).tril()
    actual = model.forward_hidden(
        hidden, condition, return_features=True, attention_mask=attention, position_ids=positions
    )
    rotary = reference.rotary_emb(ref_hidden, position_ids=positions)
    expected_hidden = ref_hidden
    for block in reference.blocks:
        expected_hidden = block(expected_hidden, rotary, ref_condition, attention)
    if kind == "continuous":
        head = reference.output
        expected = head.projection(expected_hidden) @ head.embedding.t() + head.bias
    else:
        expected = reference.output.layer(expected_hidden, ref_condition)
        if kind == "sphere":
            expected = expected[..., :-1] / (1.000001 + torch.tanh(expected[..., -1:]) * 0.999999)
    torch.testing.assert_close(actual.logits, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.hidden, expected_hidden, rtol=0, atol=0)
    actual.logits.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(hidden.grad, ref_hidden.grad, rtol=0, atol=0)
    torch.testing.assert_close(condition.grad, ref_condition.grad, rtol=0, atol=0)
    for (name, parameter), (_, ref_parameter) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        assert (parameter.grad is None) == (ref_parameter.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, ref_parameter.grad, rtol=0, atol=0)


def test_input_slot_is_chosen_in_config_without_moving_parameters():
    def config(input_target):
        return OmegaConf.create({
            "_target_": "core.models.modular.ModularDiT",
            "vocab_size": 7, "hidden_size": 16, "cond_dim": 8, "n_blocks": 1, "n_heads": 2,
            "dropout": 0.0,
            "input": {"_target_": input_target, "_partial_": True},
            "time": {"_target_": "core.models.modular.TimeConditioning", "_partial_": True},
            "output": {"_target_": "core.models.modular.AdaLNHead", "_partial_": True},
        })

    torch.manual_seed(3)
    tokens = instantiate(config("core.models.modular.TokenInput"))
    torch.manual_seed(3)
    simplex = instantiate(config("core.models.modular.SimplexInput"))
    assert isinstance(simplex, ModularDiT) and isinstance(simplex.input, SimplexInput)
    assert tokens.state_dict().keys() == simplex.state_dict().keys()
    for key, value in tokens.state_dict().items():
        assert torch.equal(value, simplex.state_dict()[key]), key

"""Behavioral contracts for interchangeable model components."""

import copy

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

from core.models.components import SimplexExpectationInput, TokenInput
from core.models.continuous_dit import ContinuousDiT
from core.models.dit import DiT
from core.models.simplex import SimplexDiT
from core.models.spherical import SphereDiT
from core.models.text_dit import TextDiT


def small(**kwargs):
    return dict(vocab_size=7, hidden_size=16, n_blocks=2, n_heads=2,
                dropout=0, **kwargs)


@pytest.mark.parametrize("kind", ["discrete", "continuous", "sphere"])
def test_shared_backbone_matches_original_execution_and_gradients(kind):
    """Bypass composed execution with the previous direct block/head path."""
    torch.manual_seed(81)
    if kind == "discrete":
        model = DiT(**small(adaLN=True))
    elif kind == "continuous":
        model = ContinuousDiT(**small())
    else:
        model = SphereDiT(**small(state_dim=7, learn_temperature_scaling=True))
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
    actual = model.forward_hidden(hidden, condition, return_features=True,
                                  attention_mask=attention, position_ids=positions)
    rotary = reference.rotary_emb(ref_hidden, position_ids=positions)
    expected_hidden = ref_hidden
    for block in reference.blocks:
        expected_hidden = block(expected_hidden, rotary, ref_condition, attention)
    if kind == "continuous":
        expected = (reference.output_projection(expected_hidden)
                    @ reference.output_embedding.t() + reference.output_bias)
    else:
        expected = reference.output_layer(expected_hidden, ref_condition)
        if kind == "sphere":
            expected = expected[..., :-1] / (
                1.000001 + torch.tanh(expected[..., -1:]) * 0.999999)
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
            torch.testing.assert_close(parameter.grad, ref_parameter.grad,
                                       rtol=0, atol=0)


def test_hydra_can_choose_simplex_input_without_a_model_subclass():
    options = small()
    composed = instantiate(OmegaConf.create({
        "_target_": "core.models.dit.DiT", **options,
        "input_adapter": {
            "_target_": "core.models.components.SimplexExpectationInput"
        },
    }))
    original = SimplexDiT(**options)
    composed.load_state_dict(original.state_dict(), strict=True)
    state = torch.randn(2, 5, 7).log_softmax(-1)
    time = torch.rand(2)
    torch.testing.assert_close(composed(state, time), original(state, time),
                               rtol=0, atol=0)
    assert isinstance(composed, TextDiT)
    assert isinstance(composed.input_adapter, SimplexExpectationInput)
    assert composed.state_dict().keys() == original.state_dict().keys()


class DoubleHead(nn.Module):
    def forward(self, hidden, condition, *, layer):
        return 2 * layer(hidden, condition)


class ShiftTime(nn.Module):
    def forward(self, coordinate, embedder):
        return torch.nn.functional.silu(embedder(coordinate + 0.1))


def test_output_and_time_components_are_injected_without_moving_parameters():
    model = DiT(**small(adaLN=True), output_adapter=DoubleHead(),
                time_adapter=ShiftTime())
    tokens, time = torch.randint(7, (2, 5)), torch.rand(2)
    hidden, condition = model.prepare_hidden(model.vocab_embed(tokens), time)
    torch.testing.assert_close(condition, torch.nn.functional.silu(model.sigma_map(time + 0.1)))
    assert not any(key.startswith(("input_adapter.", "output_adapter.", "time_adapter.",
                                  "block_execution.")) for key in model.state_dict())
    with torch.no_grad():
        model.output_layer.linear.weight.normal_()
    output = model.forward_hidden(hidden, condition, return_features=True)
    torch.testing.assert_close(output.logits, 2 * model.output_layer(output.hidden, condition))


def test_explicit_simplex_adapter_is_not_overwritten():
    adapter = TokenInput()
    model = SimplexDiT(**small(), input_adapter=adapter)
    assert model.input_adapter is adapter

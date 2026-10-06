"""Intermediate token feedback architecture and auxiliary objective."""

import copy
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import pytest
import torch

from core.modifiers.block_diffusion import BlockDiffusion
from core.modifiers.token_feedback import TokenFeedback
from core.models.dit import DiT


def model(*, self_conditioning=False, source="posterior_mean"):
    return DiT(
        vocab_size=11,
        hidden_size=16,
        cond_dim=8,
        n_blocks=3,
        n_heads=2,
        dropout=0,
        adaLN=True,
        self_conditioning=self_conditioning,
        self_conditioning_source=source,
    )


def algorithm(*overrides):
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        cfg = compose(
            config_name="config",
            overrides=["+experiment=smoke", "algorithm.length=7", *overrides],
        )
    return instantiate(cfg.algorithm)


@pytest.mark.parametrize("return_features", [False, True])
def test_zero_gate_preserves_disabled_model_and_one_backbone_pass(return_features):
    baseline = model().eval()
    enabled = copy.deepcopy(baseline)
    enabled.block_execution = __import__(
        "core.modifiers.token_feedback.model", fromlist=["TokenFeedbackBlocks"]
    ).TokenFeedbackBlocks(enabled.blocks[1], 1, 16)
    calls = []
    handles = [
        block.register_forward_hook(lambda *args: calls.append(1))
        for block in enabled.blocks
    ]
    tokens = torch.randint(0, 11, (2, 7))
    time = torch.rand(2)
    expected = baseline(tokens, time, return_features=return_features)
    actual = enabled(tokens, time, return_features=return_features)
    for handle in handles:
        handle.remove()
    # Three backbone blocks run once; the auxiliary extractor is separate.
    assert len(calls) == 3
    torch.testing.assert_close(
        actual.logits if return_features else actual,
        expected.logits if return_features else expected,
        rtol=0,
        atol=0,
    )
    if return_features:
        assert actual.auxiliary_logits.shape == (2, 7, 11)


def test_gate_gradient_then_feedback_changes_output():
    net = model().eval()
    from core.modifiers.token_feedback.model import TokenFeedbackBlocks

    net.block_execution = TokenFeedbackBlocks(net.blocks[1], 1, 16)
    torch.nn.init.normal_(net.output_layer.linear.weight, std=0.1)
    tokens, time = torch.randint(0, 11, (2, 7)), torch.rand(2)
    baseline = net(tokens, time).detach()
    net(tokens, time).square().mean().backward()
    assert net.block_execution.gate.grad.abs() > 0
    with torch.no_grad():
        net.block_execution.gate.fill_(0.5)
    assert not torch.equal(net(tokens, time), baseline)


def test_auxiliary_ce_uses_loss_mask_and_trains_extractor_after_head_bootstrap():
    wrapped = TokenFeedback(algorithm(), depth=0, loss_weight=0.7)
    tokens = torch.randint(0, 10, (2, 7))
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[:, 2:5] = True
    torch.manual_seed(9)
    with wrapped.feedback.capture() as expected_capture:
        wrapped.algorithm.loss(tokens, mask, training=True)
    expected = torch.nn.functional.cross_entropy(
        expected_capture[0].transpose(1, 2), tokens, reduction="none"
    )[mask].mean()
    torch.manual_seed(9)
    result = wrapped.loss(tokens, mask, training=True)
    torch.testing.assert_close(result["token_feedback_loss"], expected)
    result["loss"].backward()
    assert result["token_feedback_loss"].isfinite()
    assert wrapped.feedback.gate.grad is not None
    assert wrapped.feedback.extractor.attn_qkv.weight.grad is not None
    # The shared head starts at zero, so extractor gradients bootstrap only
    # after the head has taken an update.
    optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.1)
    optimizer.step()
    optimizer.zero_grad()
    wrapped.loss(tokens, mask, training=True)["loss"].backward()
    assert wrapped.feedback.extractor.attn_qkv.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("source", ["posterior_mean", "hidden"])
def test_self_conditioning_bootstrap_is_not_counted_as_auxiliary_pass(source):
    wrapped = TokenFeedback(
        algorithm(
            "self_conditioning.enabled=true",
            f"self_conditioning.source={source}",
            "self_conditioning.probability=1",
        ),
        depth=0,
    )
    tokens = torch.randint(0, 10, (2, 7))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    result = wrapped.loss(tokens, mask, training=True)
    assert result["token_feedback_loss"].isfinite()


def test_block_diffusion_preserves_auxiliary_noisy_stream_shape():
    wrapped = TokenFeedback(BlockDiffusion(algorithm(), 3), depth=0)
    tokens = torch.randint(0, 10, (2, 7))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    result = wrapped.loss(tokens, mask, training=True)
    assert result["token_feedback_loss"].isfinite()


def test_incompatible_model_has_clear_error():
    class Algorithm:
        def __init__(self):
            self.model = torch.nn.Linear(3, 3)

    with pytest.raises(TypeError, match="token embedding table"):
        TokenFeedback(Algorithm(), depth=0)


@pytest.mark.parametrize("detach", [False, True])
def test_detach_blocks_main_loss_through_auxiliary_logits_only(detach):
    from core.modifiers.token_feedback.model import TokenFeedbackBlocks

    torch.manual_seed(17)
    net = model().eval()
    net.block_execution = TokenFeedbackBlocks(net.blocks[1], 1, 16, detach=detach)
    # Enable feedback and a nonzero head so zero initialization cannot hide
    # a missing gradient path.
    torch.nn.init.normal_(net.output_layer.linear.weight, std=0.1)
    with torch.no_grad():
        net.block_execution.gate.fill_(0.5)
    tokens, time = torch.randint(0, 11, (2, 7)), torch.rand(2)
    output = net(tokens, time, return_features=True)
    main_gradient = torch.autograd.grad(
        output.logits.square().mean(), output.auxiliary_logits,
        retain_graph=True, allow_unused=True,
    )[0]
    if detach:
        assert main_gradient is None
    else:
        assert main_gradient.abs().sum() > 0
    auxiliary_loss = torch.nn.functional.cross_entropy(
        output.auxiliary_logits.transpose(1, 2), tokens
    )
    auxiliary_gradient = torch.autograd.grad(
        auxiliary_loss, output.auxiliary_logits, retain_graph=True
    )[0]
    assert auxiliary_gradient.abs().sum() > 0
    extractor_gradients = torch.autograd.grad(
        auxiliary_loss, tuple(net.block_execution.extractor.parameters()),
        allow_unused=True,
    )
    assert any(g is not None and g.abs().sum() > 0 for g in extractor_gradients)

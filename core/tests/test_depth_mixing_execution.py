from functools import partial

import pytest
import torch

from core.models.components import DiffusionBlocks
from core.models.depth_mixing.attention import DepthAttention
from core.models.depth_mixing.dca import DCA
from core.models.depth_mixing.execution import DepthMixingBlocks
from core.models.depth_mixing.mudd import MUDDDepthMixer
from core.models.layers import DDiTBlock, Rotary


@pytest.mark.parametrize("ada_ln", [False, True])
@pytest.mark.parametrize(
    "mixer,history",
    [
        (DCA, "contributions"),
        (MUDDDepthMixer, "states"),
        (partial(DepthAttention, key_dim=4), "states"),
    ],
)
def test_full_stack_initialization_preserves_baseline_and_input_gradients(
    mixer, history, ada_ln
):
    torch.manual_seed(7)
    blocks = torch.nn.ModuleList(
        [DDiTBlock(16, 2, ada_ln, 8, dropout=0) for _ in range(3)]
    )
    if ada_ln:
        for b in blocks:
            torch.nn.init.normal_(b.adaLN_modulation.weight, std=0.1)
    x = torch.randn(2, 7, 16, requires_grad=True)
    condition = torch.randn(2, 8)
    rotary = Rotary(8)(x)
    baseline = DiffusionBlocks()(x, blocks, rotary, condition=condition)
    execution = DepthMixingBlocks(16, 3, mixer, history)
    actual = execution(x, blocks, rotary, condition=condition)
    torch.testing.assert_close(actual, baseline, atol=3e-6, rtol=3e-5)
    a = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
    b = torch.autograd.grad(baseline.square().sum(), x)[0]
    torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize(
    "mixer,history",
    [
        (partial(DCA, gate_init_std=0.001), "contributions"),
        (MUDDDepthMixer, "states"),
        (partial(DepthAttention, key_dim=4), "states"),
    ],
)
def test_routing_can_train_and_reload(mixer, history):
    torch.manual_seed(11)
    blocks = torch.nn.ModuleList([DDiTBlock(16, 2, False, dropout=0) for _ in range(2)])
    execution = DepthMixingBlocks(16, 2, mixer, history)
    x = torch.randn(2, 5, 16)
    rotary = Rotary(8)(x)
    optimizer = torch.optim.Adam(execution.parameters(), lr=0.01)
    before = execution(x, blocks, rotary).detach()
    for _ in range(3):
        optimizer.zero_grad()
        loss = execution(x, blocks, rotary).square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        optimizer.step()
    after = execution(x, blocks, rotary)
    assert not torch.allclose(before, after)
    restored = DepthMixingBlocks(16, 2, mixer, history)
    restored.load_state_dict(execution.state_dict())
    torch.testing.assert_close(after, restored(x, blocks, rotary))


def test_explicit_mlp_width_cannot_be_silently_ignored():
    from core.models.sublayers import ExpandedAttention

    with pytest.raises(ValueError, match="cannot be combined"):
        DDiTBlock(16, 2, False, mlp_hidden_size=32, sublayer_factory=ExpandedAttention)

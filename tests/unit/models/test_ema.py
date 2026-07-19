import pytest
import torch

from models.ema import ExponentialMovingAverage


def _parameter(value, requires_grad=True):
    return torch.nn.Parameter(torch.tensor(value, dtype=torch.float32), requires_grad=requires_grad)


def test_rejects_invalid_decay():
    parameter = _parameter([1.0])
    with pytest.raises(ValueError, match="between 0 and 1"):
        ExponentialMovingAverage([parameter], -0.1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        ExponentialMovingAverage([parameter], 1.1)


def test_update_uses_warmup_decay_formula():
    parameter = _parameter([11.0])
    ema = ExponentialMovingAverage([parameter], decay=0.9)
    ema.shadow_params[0].zero_()

    ema.update([parameter])

    # First-update decay is min(0.9, 2 / 11), so shadow = (1 - 2/11) * 11.
    assert ema.num_updates == 1
    assert ema.shadow_params[0].item() == pytest.approx(9.0)


def test_update_without_counter_uses_configured_decay():
    parameter = _parameter([10.0])
    ema = ExponentialMovingAverage([parameter], decay=0.75, use_num_updates=False)
    ema.shadow_params[0].zero_()

    ema.update([parameter])

    assert ema.num_updates is None
    assert ema.shadow_params[0].item() == pytest.approx(2.5)


def test_multiple_updates_match_iterated_ema_recurrence():
    parameter = _parameter([0.0])
    ema = ExponentialMovingAverage([parameter], decay=0.75, use_num_updates=False)
    expected = 0.0
    for value in (4.0, -2.0, 10.0):
        parameter.data.fill_(value)
        ema.update([parameter])
        expected = 0.75 * expected + 0.25 * value
        assert ema.shadow_params[0].item() == pytest.approx(expected)


def test_store_copy_to_restore_is_symmetric_and_clears_store():
    parameter = _parameter([1.0, 2.0])
    ema = ExponentialMovingAverage([parameter], decay=0.9)
    ema.shadow_params[0].copy_(torch.tensor([8.0, 9.0]))

    ema.store([parameter])
    ema.copy_to([parameter])
    assert torch.equal(parameter, torch.tensor([8.0, 9.0]))

    ema.restore([parameter])
    assert torch.equal(parameter, torch.tensor([1.0, 2.0]))
    assert ema.collected_params == []


def test_frozen_parameters_are_ignored_consistently():
    trainable = _parameter([1.0])
    frozen = _parameter([2.0], requires_grad=False)
    ema = ExponentialMovingAverage([trainable, frozen], decay=0.5, use_num_updates=False)
    trainable.data.fill_(3.0)

    ema.store([trainable, frozen])
    ema.update([trainable, frozen])
    ema.copy_to([trainable, frozen])
    ema.restore([trainable, frozen])

    assert trainable.item() == pytest.approx(3.0)
    assert frozen.item() == pytest.approx(2.0)


def test_copy_to_copies_shadow_values():
    parameter = _parameter([1.0, 2.0])
    ema = ExponentialMovingAverage([parameter], decay=0.5)
    ema.shadow_params[0].copy_(torch.tensor([4.0, 5.0]))

    ema.copy_to([parameter])

    assert torch.equal(parameter, torch.tensor([4.0, 5.0]))


def test_state_dict_round_trip_is_independent():
    parameter = _parameter([3.0])
    source = ExponentialMovingAverage([parameter], decay=0.8)
    source.update([parameter])
    state = source.state_dict()

    restored = ExponentialMovingAverage([parameter], decay=0.1, use_num_updates=False)
    restored.load_state_dict(state)
    state["shadow_params"][0].fill_(99.0)

    assert restored.decay == pytest.approx(0.8)
    assert restored.num_updates == 1
    assert restored.shadow_params[0].item() == pytest.approx(3.0)


@pytest.mark.parametrize("operation", ["update", "copy_to", "store"])
def test_parameter_count_mismatch_raises(operation):
    first = _parameter([1.0])
    second = _parameter([2.0])
    ema = ExponentialMovingAverage([first, second], decay=0.9)

    with pytest.raises(ValueError, match="parameter count"):
        getattr(ema, operation)([first])


def test_restore_requires_matching_stored_parameters():
    first = _parameter([1.0])
    second = _parameter([2.0])
    ema = ExponentialMovingAverage([first, second], decay=0.9)
    ema.store([first, second])

    with pytest.raises(ValueError, match="parameter count"):
        ema.restore([first])

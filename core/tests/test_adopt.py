"""adopt() moves every legacy network kind into ModularDiT without copying.

The adopted network holds the legacy tensors themselves (same storage), under
the converter's names, draws no random numbers, and computes the same outputs.
"""

from hydra.utils import instantiate
import pytest
import torch

from core.models.legacy import adopt, convert_state_dict
from core.tests.test_modular_dit import CASES, inputs


@pytest.mark.parametrize("name", sorted(CASES))
def test_adopt_reuses_tensors_draws_no_rng_and_matches(name):
    config = CASES[name]
    torch.manual_seed(0)
    legacy = instantiate(config).eval()
    expected_names = convert_state_dict(legacy.state_dict(), config["_target_"])
    legacy_args, modular_args, feedback = inputs(name, config)
    with torch.no_grad():
        expected = legacy(*legacy_args, self_condition=feedback, return_features=True)
    state = torch.random.get_rng_state()
    adopted = adopt(legacy).eval()
    assert torch.equal(state, torch.random.get_rng_state())
    actual_state = adopted.state_dict(keep_vars=True)
    assert set(actual_state) == set(expected_names)
    for key, value in expected_names.items():
        assert actual_state[key].data_ptr() == value.data_ptr(), key
    with torch.no_grad():
        actual = adopted(*modular_args, self_condition=feedback, return_features=True)
    assert torch.equal(expected.logits, actual.logits)
    assert torch.equal(expected.hidden, actual.hidden)


def test_views_are_only_bound_to_token_networks():
    legacy = instantiate(CASES["continuous"])
    with pytest.raises(ValueError, match="only the token DiT"):
        adopt(legacy, views=[object()])

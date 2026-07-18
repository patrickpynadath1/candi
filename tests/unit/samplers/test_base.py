import pytest

from samplers.autoregressive import AutoregressiveSampler
from samplers.base import SamplingRequest
from samplers.runner import run_sampler

from .helpers import ModelBase


class ARContractModel(ModelBase):
    sampling_family = "autoregressive"
    sampling_capabilities = frozenset({"next_token_logits"})

    def next_token_logits(self, _tokens):
        raise AssertionError("validation tests should fail before sampling")


def test_rejects_incompatible_algorithm_family_before_sampling():
    model = ARContractModel()
    model.sampling_family = "uniform_diffusion"

    with pytest.raises(ValueError, match="not compatible"):
        run_sampler(
            AutoregressiveSampler(use_float64=False, temperature=1),
            model,
            SamplingRequest(num_samples=1),
        )


def test_reports_missing_declared_capability():
    model = ARContractModel()
    model.sampling_capabilities = frozenset()

    with pytest.raises(ValueError, match="next_token_logits"):
        run_sampler(
            AutoregressiveSampler(use_float64=False, temperature=1),
            model,
            SamplingRequest(num_samples=1),
        )


def test_verifies_method_behind_advertised_capability():
    model = ARContractModel()
    model.next_token_logits = None

    with pytest.raises(TypeError, match="next_token_logits"):
        run_sampler(
            AutoregressiveSampler(use_float64=False, temperature=1),
            model,
            SamplingRequest(num_samples=1),
        )

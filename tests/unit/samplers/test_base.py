import pytest
import torch

from samplers.autoregressive import AutoregressiveSampler
from samplers.base import Sampler, SamplerSpec, SamplingRequest, SamplingResult
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


class RecordingSampler(Sampler):
    spec = SamplerSpec("recording", frozenset({"test"}), frozenset())

    def sample(self, model, request):
        return SamplingResult(torch.zeros(request.num_samples, 1, dtype=torch.long))


class RecordingModel:
    sampling_family = "test"
    sampling_capabilities = frozenset()


@pytest.mark.parametrize(
    ("sampling_request", "message"),
    [
        (SamplingRequest(num_samples=0), "num_samples"),
        (SamplingRequest(num_samples=1, num_steps=0), "num_steps"),
        (
            SamplingRequest(
                num_samples=1, prompt_tokens=torch.zeros(1, 2, dtype=torch.long)
            ),
            "provided together",
        ),
        (
            SamplingRequest(
                num_samples=1,
                prompt_tokens=torch.zeros(1, 2, dtype=torch.long),
                prompt_mask=torch.zeros(1, 3),
            ),
            "matching shapes",
        ),
        (
            SamplingRequest(
                num_samples=2,
                prompt_tokens=torch.zeros(1, 2, dtype=torch.long),
                prompt_mask=torch.zeros(1, 2),
            ),
            "batch size",
        ),
    ],
)
def test_run_sampler_rejects_invalid_requests_before_model_execution(
    sampling_request, message
):
    with pytest.raises(ValueError, match=message):
        run_sampler(RecordingSampler(), RecordingModel(), sampling_request)

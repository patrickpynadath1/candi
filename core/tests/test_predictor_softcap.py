"""Softcapping is trainable parameterization, never a substitute for constraints."""

import pytest
import torch

from core.parameterizations.plug_in import PlugInParameterization
from core.parameterizations.posterior import PosteriorParameterization
from core.parameterizations.utils import softcap_logits


@pytest.mark.parametrize(
    "predictor_type", [PosteriorParameterization, PlugInParameterization]
)
def test_softcap_matches_reference_forward_and_gradient(predictor_type):
    logits = torch.tensor([[[8.0, -2.0, 1.0]]], dtype=torch.float64, requires_grad=True)
    predictor = predictor_type(softcap=2.0, temperature=0.1, top_k=1, top_p=0.1)
    if predictor_type is PosteriorParameterization:
        prediction = predictor.training_posterior(logits, None)
    else:
        prediction = predictor.training_plugin(logits)
    reference = (2.0 * torch.tanh(logits / 2.0)).log_softmax(-1)
    torch.testing.assert_close(prediction, reference)
    actual_gradient = torch.autograd.grad(
        -prediction[..., 1].sum(), logits, retain_graph=True
    )[0]
    expected_gradient = torch.autograd.grad(-reference[..., 1].sum(), logits)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert torch.isfinite(prediction).all()  # Inference filters never reached training.


@pytest.mark.parametrize(
    "predictor_type", [PosteriorParameterization, PlugInParameterization]
)
def test_sampling_caps_before_temperature_and_filters(predictor_type):
    logits = torch.tensor([[[8.0, -2.0, 1.0]]], dtype=torch.float64)
    predictor = predictor_type(softcap=2.0, temperature=0.5, top_k=2)
    values = 2.0 * torch.tanh(logits / 2.0) / 0.5
    values[..., 1] = -torch.inf
    expected = values.log_softmax(-1)
    if predictor_type is PosteriorParameterization:
        actual = predictor.sampling_posterior(logits, None)
        greedy = predictor.greedy_posterior(logits, None)
    else:
        actual = predictor.sampling_plugin(logits)
        greedy = predictor.greedy_plugin(logits)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        greedy.exp(), torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float64)
    )


def test_softcap_does_not_restore_excluded_mask_or_override_observations():
    predictor = PosteriorParameterization(mask_index=2, softcap=1.0)
    logits = torch.tensor([[[2.0, 1.0, 100.0], [2.0, 1.0, 100.0]]])
    observed = torch.tensor([[2, 1]])
    for method in (
        predictor.training_posterior,
        predictor.sampling_posterior,
        predictor.greedy_posterior,
    ):
        probabilities = method(logits, observed).exp()
        assert (probabilities[..., 2] == 0).all()
        torch.testing.assert_close(probabilities[0, 1], torch.tensor([0.0, 1.0, 0.0]))
        torch.testing.assert_close(probabilities.sum(-1), torch.ones((1, 2)))


@pytest.mark.parametrize(
    "predictor_type", [PosteriorParameterization, PlugInParameterization]
)
@pytest.mark.parametrize("cap", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_softcap_rejected(predictor_type, cap):
    with pytest.raises(ValueError, match="softcap"):
        predictor_type(softcap=cap)


def test_disabled_softcap_preserves_identity():
    logits = torch.randn(2, 3)
    assert softcap_logits(logits, None) is logits

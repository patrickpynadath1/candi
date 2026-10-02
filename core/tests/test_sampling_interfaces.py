"""Regression tests for posterior-owned shaping and sampler-owned orchestration."""

from unittest.mock import patch

import pytest
import torch

from core.parameterizations.posterior import PosteriorParameterization
from core.parameterizations.plug_in import PlugInParameterization
from core.parameterizations.utils import filter_top_k, filter_top_p, greedy_projection
from core.samplers.ancestral import AncestralSampler
from core.tests.test_mdlm import oracle


def test_filter_helpers_preserve_input_and_keep_the_nucleus_crossing_token():
    logits = torch.tensor([0.6, 0.3, 0.1]).log()
    before = logits.clone()
    assert filter_top_k(logits, 1).isfinite().tolist() == [True, False, False]
    assert filter_top_p(logits, 0.7).isfinite().tolist() == [True, True, False]
    torch.testing.assert_close(filter_top_p(logits, 1), logits)
    torch.testing.assert_close(filter_top_k(logits, 0), logits)
    torch.testing.assert_close(
        greedy_projection(logits).exp(), torch.tensor([1.0, 0.0, 0.0])
    )
    torch.testing.assert_close(logits, before)


@pytest.mark.parametrize("commitment", ["greedy", "sample"])
def test_posterior_owns_commitment_policy_and_greedy_denoising_is_explicit(commitment):
    posterior = PosteriorParameterization(
        3, temperature=0.5, top_k=2, commitment=commitment
    )
    state = torch.tensor([[3, 1]])
    logits = torch.tensor([[[1.0, 2.0, 3.0, 99.0], [3.0, 2.0, 1.0, 99.0]]])
    prediction = posterior.sampling_posterior(logits, state)
    probabilities = posterior.probabilities(prediction, apply_commitment=True)
    deterministic = posterior.commitment == "greedy"
    assert deterministic == (commitment == "greedy")
    assert probabilities[0, 1, 1] == 1
    if deterministic:
        assert probabilities[0, 0, 2] == 1
    else:
        torch.testing.assert_close(probabilities, prediction.exp())
    assert posterior.entropy(prediction)[0, 0] > 0
    greedy = posterior.greedy_posterior(logits, state).exp()
    assert greedy.tolist() == [[[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]


@pytest.mark.parametrize("final", ["none", "greedy", "ancestral"])
def test_ancestral_sampler_orchestrates_process_and_final_denoising(final):
    algorithm = oracle()
    prompt = torch.zeros(1, 5, dtype=torch.long)
    with (
        patch.object(
            algorithm.process, "reverse", wraps=algorithm.process.reverse
        ) as reverse,
        patch.object(
            algorithm.process, "commit", wraps=algorithm.process.commit
        ) as commit,
        patch.object(
            algorithm, "greedy_prediction", wraps=algorithm.greedy_prediction
        ) as greedy,
    ):
        _, metadata = algorithm.generate(
            AncestralSampler(2, final_denoise=final), prompt, prompt.bool()
        )
    assert reverse.call_count == 2 + (final == "ancestral")
    assert commit.call_count == (final == "greedy")
    assert greedy.call_count == (final == "greedy")
    assert metadata["nfe"] == 2 + (final != "none")


def test_eb_reuses_prediction_for_entropy_and_commitment_without_extra_model_calls():
    from core.samplers.entropy_bounded import EntropyBoundedSampler

    class Uncertain(torch.nn.Module):
        def forward(self, tokens, conditioning):
            logits = torch.zeros(*tokens.shape, 4)
            logits[..., 1] = 0.1
            return logits

    algorithm = oracle()
    algorithm.model = Uncertain()
    prompt = torch.zeros(1, 5, dtype=torch.long)
    calls = []
    hook = algorithm.model.register_forward_hook(lambda *args: calls.append(1))
    with (
        patch.object(
            algorithm.predictor, "entropy", wraps=algorithm.predictor.entropy
        ) as entropy,
        patch.object(
            algorithm.predictor,
            "probabilities",
            wraps=algorithm.predictor.probabilities,
        ) as commitment,
    ):
        result, metadata = algorithm.generate(
            EntropyBoundedSampler(0, 5), prompt, prompt.bool()
        )
    hook.remove()
    assert len(calls) == metadata["nfe"] == entropy.call_count == commitment.call_count
    for entropy_call, commitment_call in zip(
        entropy.call_args_list, commitment.call_args_list
    ):
        assert entropy_call.args[0] is commitment_call.args[0]
    assert metadata["nfe"] == 5
    assert result.tolist() == [[1, 1, 1, 1, 1]]
    assert prompt.eq(0).all()


@pytest.mark.parametrize(
    "parameterization", [PosteriorParameterization, PlugInParameterization]
)
@pytest.mark.parametrize("commitment", ["sample", "greedy"])
def test_probability_conversion_preserves_precision_and_unprojected_entropy(
    parameterization, commitment
):
    predictor = parameterization(commitment=commitment)
    prediction = torch.tensor([[[0.2, 0.5, 0.3]]], dtype=torch.float64).log()
    original = prediction.clone()
    entropy = predictor.entropy(prediction)
    full = predictor.probabilities(prediction)
    projected = predictor.probabilities(prediction, apply_commitment=True)
    torch.testing.assert_close(full, prediction.exp())
    assert projected.dtype == torch.float64
    if commitment == "greedy":
        assert projected.tolist() == [[[0.0, 1.0, 0.0]]]
    else:
        torch.testing.assert_close(projected, full)
    torch.testing.assert_close(prediction, original)
    torch.testing.assert_close(predictor.entropy(prediction), entropy)
    assert entropy.item() > 0

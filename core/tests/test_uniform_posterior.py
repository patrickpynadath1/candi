"""Posterior uniform bridges differ from plugging weights into Bayes' rule."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from core.processes.uniform import UniformProcess
from core.processes.uniform_posterior import PosteriorUniformProcess
from core.modifiers.hybrid.views import ViewPoint, DiffusionView
from core.samplers.hybrid_updates import DiscretePosteriorUpdate


@pytest.mark.parametrize(
    "at,ass", [(0.0, 0.7), (0.2, 0.7), (0.9, 1.0), (0.4, 0.4), (0.0, 0.0), (1.0, 1.0)]
)
def test_enumerated_bayes_bridge(at, ass):
    size = 4
    posterior = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]], dtype=torch.float64)
    state = torch.tensor([[2]])
    actual = PosteriorUniformProcess(size).reverse_probabilities(
        state,
        posterior,
        torch.tensor([at], dtype=torch.float64),
        torch.tensor([ass], dtype=torch.float64),
    )
    expected = torch.zeros(size, dtype=torch.float64)
    if at == ass:
        expected[2] = 1
    else:
        for clean in range(size):
            denominator = at * (clean == 2) + (1 - at) / size
            for previous in range(size):
                forward_s = ass * (previous == clean) + (1 - ass) / size
                transition = at / ass * (previous == 2) + (1 - at / ass) / size
                expected[previous] += (
                    posterior[0, 0, clean] * forward_s * transition / denominator
                )
    torch.testing.assert_close(actual[0, 0], expected)
    torch.testing.assert_close(actual.sum(-1), torch.ones(1, 1, dtype=torch.float64))
    assert (actual >= 0).all()
    if ass == 1 and at != ass:
        torch.testing.assert_close(actual, posterior)


def test_posterior_does_not_recondition_clean_prediction():
    state = torch.tensor([[1]])
    posterior = torch.tensor([[[0.2, 0.3, 0.5]]])
    at, ass = torch.tensor([0.5]), torch.tensor([1.0])
    posterior_reverse = PosteriorUniformProcess(3).reverse_probabilities(
        state, posterior, at, ass
    )
    plugin_reverse = UniformProcess(3).reverse_probabilities(state, posterior, at, ass)
    torch.testing.assert_close(posterior_reverse, posterior)
    assert not torch.allclose(posterior_reverse, plugin_reverse)


@pytest.mark.parametrize("self_conditioning", [False, True])
def test_config_training_sampling_and_unconstrained_observations(self_conditioning):
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        config = compose(
            config_name="config",
            overrides=[
                "+experiment=uniform_posterior",
                "model.hidden_size=16",
                "model.n_heads=2",
                "model.n_blocks=1",
                "model.cond_dim=8",
                "model.dropout=0",
                "sampler.steps=2",
                f"self_conditioning.enabled={str(self_conditioning).lower()}",
                "self_conditioning.source=hidden",
            ],
        )
    algo, sampler = instantiate(config.algorithm), instantiate(config.sampler)
    tokens = torch.randint(algo.process.vocab_size, (2, 6))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, 0] = False
    loss = algo.loss(tokens, mask)["loss"]
    assert loss.isfinite()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in algo.parameters())
    probabilities = algo.training_prediction(tokens, torch.full((2,), 0.4)).exp()
    assert (probabilities > 0).all()  # No absorbing-state clamp of observed tokens.
    generated, info = algo.generate(sampler, tokens, ~mask)
    assert generated.shape == tokens.shape and info["nfe"] == 3
    assert torch.equal(generated[:, 0], tokens[:, 0])
    assert generated.min() >= 0 and generated.max() < algo.process.vocab_size


def test_hybrid_branch_selects_posterior_reverse():
    process = PosteriorUniformProcess(3)
    branch = DiffusionView(process, None, None, DiscretePosteriorUpdate())
    state = torch.tensor([[2, 2]])
    posterior = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    source = ViewPoint(torch.tensor([0.5]), (torch.tensor([0.5]), None))
    destination = ViewPoint(torch.tensor([0.0]), (torch.tensor([1.0]), None))
    assert torch.equal(
        branch.update(process, None, state, posterior, source, destination),
        torch.tensor([[0, 1]]),
    )


def test_hybrid_uniform_posterior_training_sampling_and_plugin_rejection():
    from core.modifiers.hybrid.training import HybridDiffusion
    from core.modifiers.hybrid.model import HybridModel
    from core.representations.noisy_inputs import TokenRepresentation
    from core.models.dit import DiT
    from core.noise_schedules.log_linear import LogLinear
    from core.objectives.cross_entropy import CrossEntropy
    from core.parameterizations.posterior import PosteriorParameterization
    from core.samplers.hybrid import HybridSampler
    from core.time_sampling.uniform import UniformTimeSampler

    def build(process):
        branch = DiffusionView(
            process,
            LogLinear(0.001),
            TokenRepresentation(4, 16),
            DiscretePosteriorUpdate(),
        )
        backbone = DiT(4, 16, 8, 1, 2, dropout=0, adaLN=False)
        model = HybridModel(backbone, [branch])
        return HybridDiffusion(
            model,
            PosteriorParameterization(),
            UniformTimeSampler(0.001, False),
            CrossEntropy(),
            length=6,
        )

    with pytest.raises(TypeError, match="uniform_posterior"):
        build(UniformProcess(4))
    algo = build(PosteriorUniformProcess(4))
    tokens = torch.randint(4, (2, 6))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    loss = algo.loss(tokens, mask)["loss"]
    assert loss.isfinite()
    loss.backward()
    generated, _ = algo.generate(HybridSampler(steps=2), tokens, ~mask)
    assert generated.shape == tokens.shape


def test_entropy_bounded_rejected_update_keeps_full_posterior():
    from types import SimpleNamespace
    from core.parameterizations.posterior import PosteriorParameterization
    from core.samplers.uniform_entropy_bounded import UniformEntropyBoundedSampler

    process = PosteriorUniformProcess(3)
    probabilities = torch.tensor([[[0.2, 0.3, 0.5]]])
    captured = []
    original = process.reverse

    def reverse(state, posterior, keep_t, keep_s):
        captured.append(posterior)
        return original(state, posterior, keep_t, keep_s)

    process.reverse = reverse
    algo = SimpleNamespace(
        process=process,
        predictor=PosteriorParameterization(commitment="greedy"),
        schedule=lambda t: (1 - t * 0.999, None),
        sampling_prediction=lambda state, keep: probabilities.log(),
    )
    tokens = torch.zeros(1, 1, dtype=torch.long)
    UniformEntropyBoundedSampler(entropy_bound=10, steps=1).sample(
        algo, tokens, tokens.bool()
    )
    torch.testing.assert_close(captured[0], probabilities.double())


@pytest.mark.parametrize("freeze", [False, True])
def test_uniform_eb_revisits_previously_committed_positions(freeze):
    from types import SimpleNamespace
    from core.parameterizations.posterior import PosteriorParameterization
    from core.samplers.uniform_entropy_bounded import UniformEntropyBoundedSampler

    process = PosteriorUniformProcess(3)
    states = []
    transitions = []
    # Step one selects position 1; step two selects position 2. Position 1
    # must still receive the reverse transition after its earlier commitment.
    predictions = [
        torch.tensor([[[.2, .3, .5], [.001, .998, .001], [.3, .3, .4]]]),
        torch.tensor([[[.2, .3, .5], [.3, .3, .4], [.001, .001, .998]]]),
    ]

    def predict(state, keep):
        states.append(state.clone())
        return predictions[len(states) - 1].log()

    def reverse(state, probabilities, keep_t, keep_s):
        transitions.append(state.clone())
        return torch.zeros_like(state)

    process.reverse = reverse
    algo = SimpleNamespace(
        process=process, predictor=PosteriorParameterization(),
        schedule=lambda t: (1 - t * .999, None), sampling_prediction=predict,
    )
    tokens = torch.tensor([[2, 0, 0]])
    sample, info = UniformEntropyBoundedSampler(steps=2, freeze=freeze).sample(
        algo, tokens, torch.tensor([[True, False, False]])
    )
    assert len(states) == len(transitions) == 2
    assert states[1][0, 1] == 1  # previously committed token enters next step
    assert sample.tolist() == [[2, 1 if freeze else 0, 2]]
    assert info["completed_per_sample"] == [freeze]
    assert all(state[0, 0] == 2 for state in states)
    assert info['nfe_per_sample'] == [2]



def test_uniform_eb_stops_only_on_simultaneous_acceptance():
    from types import SimpleNamespace
    from core.parameterizations.posterior import PosteriorParameterization
    from core.samplers.uniform_entropy_bounded import UniformEntropyBoundedSampler

    calls = []
    def predict(state, keep):
        calls.append(state.clone())
        probabilities = [
            [[[.02, .96, .02], [.3, .3, .4]]],
            [[[.3, .3, .4], [.02, .02, .96]]],
            [[[.001, .001, .998], [.001, .998, .001]]],
        ][len(calls) - 1]
        return torch.tensor(probabilities).log()

    algo = SimpleNamespace(
        process=PosteriorUniformProcess(3), predictor=PosteriorParameterization(),
        schedule=lambda t: (1 - t * .999, None), sampling_prediction=predict,
    )
    tokens = torch.zeros(1, 2, dtype=torch.long)
    sample, info = UniformEntropyBoundedSampler(entropy_bound=.1, steps=5).sample(
        algo, tokens, tokens.bool()
    )
    assert info['accepted_per_step'] == [[1], [1], [2]]
    assert sample.tolist() == [[2, 1]]
    assert info['nfe_per_sample'] == [3]
    assert info['completed_per_sample'] == [True]
    _, empty = UniformEntropyBoundedSampler(steps=2).sample(algo, tokens, ~tokens.bool())
    assert empty['nfe_per_sample'] == [0]

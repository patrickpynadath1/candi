"""DFM rate mathematics and integration with the existing sampler policies."""

from types import SimpleNamespace

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from core.modifiers.hybrid.views import ViewPoint
from core.noise_schedules.log_linear import LogLinear
from core.processes.absorbing import AbsorbingProcess
from core.processes.uniform import UniformProcess
from core.processes.uniform_posterior import PosteriorUniformProcess
from core.samplers.discrete_flow import DiscreteFlowTransition
from core.tests.test_sampler_equivalence import build_algorithm


def point(time, keep, derivative):
    return ViewPoint(
        torch.tensor([time]), (torch.tensor([keep]), torch.tensor([derivative]))
    )


@pytest.mark.parametrize("masked", [False, True])
def test_empirical_transition_matches_conditional_ctmc_formula(masked):
    torch.manual_seed(923)
    count = 100_000
    process = AbsorbingProcess(3) if masked else PosteriorUniformProcess(3)
    view = SimpleNamespace(process=process)
    posterior = torch.tensor([0.2, 0.3, 0.5])
    endpoints = torch.distributions.Categorical(probs=posterior).sample((1, count))
    state = torch.full_like(endpoints, 3 if masked else 0)
    # Nonlinear k(t)=(1-t)^2; rate must use k'(t), not a finite difference.
    source, destination = point(0.6, 0.16, -0.8), point(0.4, 0.36, -1.2)
    result = DiscreteFlowTransition()(view, state, None, endpoints, source, destination)
    jump = -torch.expm1(torch.tensor(-0.2 * 0.8 / 0.84))
    expected = (
        torch.cat((posterior * jump, (1 - jump).reshape(1)))
        if masked
        else posterior * jump
    )
    if not masked:
        expected[0] += 1 - jump
    observed = torch.bincount(result.flatten(), minlength=len(expected)) / count
    torch.testing.assert_close(observed, expected, atol=0.006, rtol=0)


def test_absorbing_committed_tokens_and_clean_endpoint_are_preserved():
    rule = DiscreteFlowTransition()
    view = SimpleNamespace(process=AbsorbingProcess(3))
    state = torch.tensor([[0, 1, 2]])
    endpoint = torch.tensor([[2, 2, 0]])
    result = rule(
        view, state, None, endpoint, point(0.5, 0.5, -1.0), point(0.0, 1.0, -1.0)
    )
    assert torch.equal(result, state)
    view.process = PosteriorUniformProcess(3)
    result = rule(
        view, state, None, endpoint, point(0.0, 1.0, -1.0), point(0.0, 1.0, -1.0)
    )
    assert torch.equal(result, state)


def test_invalid_paths_fail_before_sampling():
    rule = DiscreteFlowTransition()
    with pytest.raises(TypeError, match="plug-in"):
        rule.validate(
            [SimpleNamespace(process=UniformProcess(3), schedule=LogLinear())], "none"
        )
    with pytest.raises(TypeError, match="keep-rate"):
        rule.validate(
            [SimpleNamespace(process=AbsorbingProcess(3), schedule=object())], "none"
        )
    with pytest.raises(ValueError, match="monotonically"):
        rule(
            SimpleNamespace(process=AbsorbingProcess(3)),
            torch.tensor([[3]]),
            None,
            torch.tensor([[0]]),
            point(0.4, 0.6, -1.0),
            point(0.5, 0.5, -1.0),
        )


@pytest.mark.parametrize("family", ["masked", "uniform"])
@pytest.mark.parametrize("sc", ["none", "posterior_mean"])
@pytest.mark.parametrize("mode", ["sample", "greedy"])
def test_dfm_config_uses_shared_loop_with_prompts_sc_and_fixed_nfe(family, sc, mode):
    algo = build_algorithm(family, None, sc)
    config = OmegaConf.load("configs/sampler/dfm.yaml")
    config.steps = 3
    config.tokens.mode = mode
    sampler = instantiate(config)
    tokens = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    prompt = torch.tensor([[True] * 4, [True, False, False, False]])
    calls = []
    with algo.model.register_forward_pre_hook(lambda module, args: calls.append(args)):
        output, info = algo.generate(sampler, tokens, prompt)
    assert torch.equal(output[prompt], tokens[prompt])
    assert ((output >= 0) & (output < 12)).all()
    assert len(calls) == info["nfe"] == 4
    assert sampler.steps_for_nfe(128) == 127


def test_dfm_composes_over_multiple_discrete_views():
    algo = build_algorithm("masked", "uniform", "none")
    config = OmegaConf.load("configs/sampler/dfm.yaml")
    config.steps = 2
    output, info = algo.generate(
        instantiate(config),
        torch.ones(2, 4, dtype=torch.long),
        torch.zeros(2, 4, dtype=torch.bool),
    )
    assert output.shape == (2, 4)
    assert info["nfe"] == 3

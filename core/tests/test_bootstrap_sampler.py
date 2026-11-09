from types import SimpleNamespace
import torch
import pytest
from core.modifiers.self_conditioning.session import prediction_session
from core.samplers.self_conditioning import BootstrapSession, BootstrapSelfConditioningSampler
from core.samplers.heun import HeunSampler

class FakeAlgorithm:
    prediction = SimpleNamespace(enabled=True, source='hidden')
    def __init__(self):
        self.seen = []
    def sampling_prediction(self, value, trajectory):
        self.seen.append(trajectory.conditioning)
        trajectory.update(value)
        return value
    greedy_prediction = sampling_prediction

def test_bootstrap_refreshes_at_every_solver_stage():
    algo = FakeAlgorithm()
    session = BootstrapSession(algo)
    session.prompt_mask = None
    assert prediction_session(session) is session
    with session.step():
        session.sampling_prediction(2)
        session.sampling_prediction(7)
    assert algo.seen == [None, 2, None, 7]

def test_bootstrap_heun_budget():
    sampler = BootstrapSelfConditioningSampler(HeunSampler())
    assert sampler.steps_for_nfe(66) == 16
    assert sampler.steps_for_nfe(258) == 64
    with pytest.raises(ValueError):
        sampler.steps_for_nfe(4)

def test_bootstrap_reports_actual_network_calls():
    class Inner:
        def sample(self, algo, tokens, mask):
            algo = prediction_session(algo)
            for i in range(self.steps + 1):
                algo.sampling_prediction(i)
            return tokens, {'nfe': self.steps + 1, 'nfe_per_sample':[self.steps + 1]}
    algo = FakeAlgorithm()
    wrapper = BootstrapSelfConditioningSampler(Inner(), steps=3)
    _, info = wrapper.sample(algo, torch.zeros(1,2), torch.zeros(1,2))
    assert len(algo.seen) == info['nfe'] == 8
    assert info['nfe_per_sample'] == [8]

@pytest.mark.parametrize("kind", ["euler", "heun"])
def test_bootstrap_updates_cached_solver_grid(kind):
    from core.samplers.euler import EulerSampler
    inner = (EulerSampler if kind == "euler" else HeunSampler)(steps=32)
    class GridProbe(type(inner)):
        def sample(self, algo, tokens, mask):
            calls = (len(self.time_grid(tokens.device)) - 1) * (2 if kind == "heun" else 1) + 1
            return tokens, {"nfe": calls, "nfe_per_sample": [calls]}
    inner.__class__ = GridProbe
    wrapper = BootstrapSelfConditioningSampler(inner)
    wrapper.steps = wrapper.steps_for_nfe(10)
    _, info = wrapper.sample(FakeAlgorithm(), torch.zeros(1, 2), torch.zeros(1, 2))
    assert info["nfe"] == 10
    assert inner.time_grid.steps == wrapper.steps

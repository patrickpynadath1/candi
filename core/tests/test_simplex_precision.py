"""FP32 simplex math preserves distributions, exact faces and state precision."""
import pytest
import torch
from core.processes.dirichlet import DirichletProcess
from core.noise_schedules.simplex import SimplexSchedule
from core.processes.simplex_math import log_dirichlet_sample


@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('churn', [0., .2, 1.])
def test_process_precision_and_reverse_moments(precision, churn):
    torch.manual_seed(42)
    process = DirichletProcess(3, precision=precision)
    schedule = SimplexSchedule()
    tokens = torch.zeros(12000, 1, dtype=torch.long)
    source = schedule(torch.full((12000,), .85, dtype=torch.float64))
    destination = schedule(torch.full((12000,), .25, dtype=torch.float64))
    state = process.corrupt(tokens, source)
    reverse = process.reverse(state, tokens, destination, source, churn)
    prior = process.prior(2, 3, schedule(torch.ones(2, dtype=torch.float64)))
    dtype = getattr(torch, precision)
    assert state.dtype == reverse.dtype == prior.dtype == process.encode(tokens[:2]).dtype == dtype
    assert torch.isfinite(state).all() and torch.isfinite(reverse).all()
    p = reverse.exp()[:, 0]
    expected = torch.tensor([1 - .25 + .25/3, .25/3, .25/3], dtype=dtype)
    variance = expected * (1 - expected) / (destination.concentration[0].to(dtype) + 1)
    torch.testing.assert_close(p.mean(0), expected, atol=.015, rtol=0)
    torch.testing.assert_close(p.var(0), variance, atol=.01, rtol=0)
    torch.testing.assert_close(reverse.logsumexp(-1), torch.zeros_like(reverse[..., 0]), atol=2e-6, rtol=0)


def test_float32_tiny_concentrations_and_exact_faces():
    beta = torch.tensor([[1e-8, 2e-8, 0.], [0., 2., 0.]], dtype=torch.float32)
    p = log_dirichlet_sample(beta)
    assert p.dtype == torch.float32
    assert torch.isfinite(p[0, :2]).all()
    assert torch.isneginf(p[:, 2]).all()
    torch.testing.assert_close(p.exp()[1], torch.tensor([0., 1., 0.]))
    torch.testing.assert_close(p.logsumexp(-1), torch.zeros(2), atol=2e-6, rtol=0)


def test_hybrid_clamp_does_not_promote_float32():
    from core.tests.test_hybrid_composition import compose_hybrid
    from core.modifiers.hybrid.views import prior, clamp
    algo, _ = compose_hybrid('simplex', 'masked')
    view = algo.views[0]
    tokens = torch.randint(0, 12, (2, 6))
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[:, :2] = True
    point = algo.points(torch.ones(2))[0]
    state = prior(view, tokens, point, mask)
    assert state.dtype == torch.float32
    assert clamp(view, state, tokens, mask).dtype == torch.float32
    assert torch.equal(state[:, :2].argmax(-1), tokens[:, :2])


def test_float64_checkpoint_loads_into_float32_reference():
    old = DirichletProcess(3, precision='float64')
    new = DirichletProcess(3, precision='float32')
    new.load_state_dict(old.state_dict(), strict=True)
    assert new.reference.dtype == torch.float32
    torch.testing.assert_close(new.reference.double(), old.reference)


def test_float32_full_vocabulary_log_normalization():
    torch.manual_seed(7)
    process = DirichletProcess(49153, precision='float32')
    tokens = torch.tensor([[0, 49152], [31, 42]])
    state = process.corrupt(tokens, SimplexSchedule()(torch.tensor([.001, .999])))
    assert state.dtype == torch.float32 and torch.isfinite(state).all()
    torch.testing.assert_close(state.logsumexp(-1), torch.zeros(2, 2), atol=2e-6, rtol=0)

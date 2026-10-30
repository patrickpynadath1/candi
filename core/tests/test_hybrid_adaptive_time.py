"""Adaptive continuous clocks must not warp the discrete corruption clock."""
import pytest
import torch
from core.tests.test_hybrid_diffusion import make_algorithm
from core.time_sampling.loss_derivative import LossDerivativeTimeSampler


def algorithm():
    algo = make_algorithm(self_conditioning=True)
    algo.time_sampler = LossDerivativeTimeSampler(adaptive=True, grid_size=101)
    algo.adaptive_time_views = (1,)  # Test fixture orders discrete then Gaussian.
    with torch.no_grad():
        algo.time_sampler.cdf.copy_(algo.time_sampler.grid.square())
    return algo


@pytest.mark.parametrize('training', [False, True])
def test_only_continuous_points_are_warped(training):
    algo = algorithm()
    time = torch.tensor([0.04, 0.25, 0.64])
    discrete, continuous = algo.points(time, training=training)
    torch.testing.assert_close(discrete.time, time)
    torch.testing.assert_close(continuous.time, torch.tensor([0.2, 0.5, 0.8]))


def test_loss_draws_uniform_and_observes_continuous_time(monkeypatch):
    algo = algorithm()
    uniform = torch.tensor([0.04, 0.25, 0.64])
    seen = []
    monkeypatch.setattr(algo.time_sampler.uniform, 'sample', lambda *args, **kwargs: uniform)
    monkeypatch.setattr(algo.time_sampler, 'sample', lambda *args, **kwargs: pytest.fail('global adaptive draw'))
    monkeypatch.setattr(algo.time_sampler, 'observe', lambda t, *args: seen.append(t.clone()))
    original = algo.training_prediction
    def prediction(states, points, time):
        torch.testing.assert_close(time, uniform)
        torch.testing.assert_close(points[0].time, uniform)
        torch.testing.assert_close(points[1].time, torch.tensor([0.2, 0.5, 0.8]))
        return original(states, points, time)
    monkeypatch.setattr(algo, 'training_prediction', prediction)
    tokens = torch.randint(0, 5, (3, 7))
    result = algo.loss(tokens, torch.ones_like(tokens, dtype=torch.bool), training=True)
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    torch.testing.assert_close(seen[0], torch.tensor([0.2, 0.5, 0.8]))


@pytest.mark.parametrize('kind', ['masked', 'uniform'])
def test_corrected_presets_select_gaussian_only(kind):
    from core.sweeps.configuration import compose
    cfg = compose(f'tinygsm_{kind}_gaussian_sc_adaptive')
    assert list(cfg.algorithm.adaptive_time_views) == [0]
    assert cfg.model.views.first.process._target_.endswith('.GaussianProcess')
    assert cfg.time_sampler.adaptive


def test_generation_uses_same_clock_mapping():
    from core.samplers.hybrid import HybridSampler
    algo = algorithm().eval()
    tokens = torch.randint(0, 5, (2, 7))
    prompt = torch.zeros_like(tokens, dtype=torch.bool)
    prompt[:, :2] = True
    generated, _ = algo.generate(HybridSampler(steps=2), tokens, prompt)
    assert torch.equal(generated[:, :2], tokens[:, :2])
    assert generated.min() >= 0 and generated.max() < 5


@pytest.mark.parametrize('kind', ['masked', 'uniform'])
def test_simplex_hybrid_local_time_training_and_generation(kind, monkeypatch):
    from hydra.utils import instantiate
    from core.sweeps.configuration import compose
    from core.training.module import TrainingModule
    cfg = compose(f'tinygsm_simplex_{kind}_sc')
    assert list(cfg.algorithm.adaptive_time_views) == [0]
    assert cfg.model.views.first.process._target_.endswith('.DirichletProcess')
    cfg.task = {'_target_': 'core.tests.test_entrypoint.TinyTask', 'vocab_size': 32, 'length': 8}
    backbone = cfg.model.backbone
    backbone.hidden_size = 32
    backbone.cond_dim = 16
    backbone.n_heads = 4
    backbone.n_blocks = 1
    backbone.dropout = 0
    cfg.sampler.steps = 2
    algo = TrainingModule(cfg).algorithm
    with torch.no_grad():
        n = algo.time_sampler.bins
        edges = torch.linspace(0, 1, n + 1)
        widths = edges.square().diff()
        logits = (widths * (1 + n * algo.time_sampler.min_bin_size) - algo.time_sampler.min_bin_size).log()
        algo.time_sampler.ema_logits[1].copy_(logits)
        algo.time_sampler.ready.fill_(True)
    time = torch.tensor([0.04, 0.25, 0.64])
    eps = algo.time_sampler.uniform.epsilon
    expected = eps + (1 - eps) * ((time - eps) / (1 - eps)).sqrt()
    for training in [True, False]:
        continuous, discrete = algo.points(time, training=training)
        torch.testing.assert_close(discrete.time, time)
        torch.testing.assert_close(continuous.time, expected, atol=2e-4, rtol=2e-4)
    monkeypatch.setattr(algo.time_sampler.uniform, 'sample', lambda *a, **kw: time)
    seen = []
    monkeypatch.setattr(algo.time_sampler, 'observe', lambda t, *a: seen.append(t.clone()))
    tokens = torch.randint(0, 31, (3, 8))
    target = torch.ones_like(tokens, dtype=torch.bool)
    target[:, :2] = False
    result = algo.loss(tokens, target, training=True)
    assert result['loss'].isfinite()
    result['loss'].backward()
    assert all(p.grad.isfinite().all() for p in algo.parameters() if p.grad is not None)
    torch.testing.assert_close(seen[0], expected, atol=2e-4, rtol=2e-4)
    algo.eval()
    generated, _ = algo.generate(instantiate(cfg.sampler), tokens, ~target)
    assert torch.equal(generated[:, :2], tokens[:, :2])
    assert generated.min() >= 0 and generated.max() < 32


@pytest.mark.parametrize("continuous", ["gaussian", "simplex"])
@pytest.mark.parametrize("discrete", ["masked", "uniform"])
@pytest.mark.parametrize("cdf_source", ["plateau", "refit"])
def test_flat_cdf_preserves_prior_and_first_reverse_endpoint(
    continuous, discrete, cdf_source, monkeypatch
):
    from core.tests.test_hybrid_composition import compose_hybrid

    algo, sampler = compose_hybrid(continuous, discrete, feedback="hidden")
    algo.eval()
    algo.time_sampler = LossDerivativeTimeSampler(adaptive=True, grid_size=101)
    algo.adaptive_time_views = (0,)
    clock = algo.time_sampler
    with torch.no_grad():
        if cdf_source == "plateau":
            # Both tails are flat; interior quantiles must still be adaptive.
            clock.cdf.copy_(((clock.grid - 0.2) / 0.6).clamp(0, 1))
        else:
            # A normal fit to a saturating loss curve also produces a flat tail.
            times = torch.linspace(0.001, 0.999, 1000)
            clock.observe(times, times.clamp(max=0.8)[:, None], None, step=1000)
            assert clock.ready
    assert clock(torch.ones(1)).item() < 1  # Reproduce the old failure condition.
    time = torch.tensor([0., 0.25, 0.75, 1.])
    for training in (False, True):
        points = algo.points(time, training=training)
        torch.testing.assert_close(points[0].time[[0, -1]], time[[0, -1]], rtol=0, atol=0)
        torch.testing.assert_close(points[0].time[1:-1], clock(time[1:-1]))
        torch.testing.assert_close(points[1].time, time, rtol=0, atol=0)

    view = algo.views[0]
    original_prior = view.process.prior_state
    original_update = view.update
    events = []

    def prior(clean, coefficients):
        if continuous == "gaussian":
            # The rank schedule has float32 interpolation roundoff at t=1.
            torch.testing.assert_close(coefficients.alpha, torch.zeros_like(coefficients.alpha), rtol=0, atol=1e-7)
        else:
            torch.testing.assert_close(coefficients.time, torch.ones_like(coefficients.time), rtol=0, atol=0)
        events.append("prior")
        return original_prior(clean, coefficients)

    def update(process, representation, state, probabilities, source, destination, **kwargs):
        if events == ["prior"]:
            torch.testing.assert_close(source.time, torch.ones_like(source.time), rtol=0, atol=0)
            assert (destination.time < source.time).all()
            events.append("first_reverse")
        result = original_update(process, representation, state, probabilities, source, destination, **kwargs)
        assert not torch.isnan(result).any()
        return result

    monkeypatch.setattr(view.process, "prior_state", prior)
    monkeypatch.setattr(view, "update", update)
    tokens = torch.randint(0, 12, (2, 6))
    prompt = torch.zeros_like(tokens, dtype=torch.bool)
    prompt[:, :2] = True
    generated, _ = algo.generate(sampler, tokens, prompt)
    assert events == ["prior", "first_reverse"]
    assert torch.equal(generated[:, :2], tokens[:, :2])
    assert generated.min() >= 0 and generated.max() < 12

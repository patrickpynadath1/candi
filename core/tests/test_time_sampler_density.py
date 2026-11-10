"""The training time sampler is separate from the noise schedule.

A noise schedule fixes what each t means; a training time sampler decides how
often training visits each t. These tests check the contract that keeps them
separable: every sampler reports the density of its draws, adaptive noise
schedules remove that density from their fit, the objective weighting is an
explicit choice, and generation never consults the training time sampler.
"""

import math

import pytest
import torch
from hydra.utils import instantiate

from core.algorithms.adaptive import observe_training_loss
from core.noise_schedules.adaptive import CDCDAdaptiveSchedule, SFLMAdaptiveSchedule
from core.noise_schedules.gumbel import GumbelSchedule
from core.noise_schedules.log_linear import LogLinear
from core.sweeps.configuration import compose
from core.time_sampling.adaptive_allocation import AdaptiveTimeAllocation
from core.time_sampling.cdcd import CDCDTimeSampler
from core.time_sampling.density import training_time_density, weight_objective
from core.time_sampling.discrete import DiscreteTimeSampler
from core.time_sampling.fixed import FixedTimeSampler
from core.time_sampling.importance import (
    PiecewiseImportanceTimeSampler,
    StratifiedUniformTimeSampler,
)
from core.time_sampling.loss_derivative import LossDerivativeTimeSampler
from core.time_sampling.piecewise import PiecewiseTimeSampler
from core.time_sampling.uniform import UniformTimeSampler

N = 12800


# --- (d) every training time sampler reports a density matching its draws ----


def fitted_loss_derivative():
    sampler = LossDerivativeTimeSampler(
        adaptive=True, buffer_size=4096, bins=20, warmup=0, refit_every=1, momentum=0
    )
    torch.manual_seed(1)
    times = torch.rand(4096)
    sampler.observe(times, (times.square() * 3)[:, None], None, step=0)
    assert sampler.ready
    return sampler


def fitted_cdcd_time_sampler():
    sampler = CDCDTimeSampler(adaptive=True, bins=10, epsilon=1e-5)
    with torch.no_grad():
        sampler.ema_logits[0].copy_(torch.linspace(-1, 1, 10))
        sampler.ema_logits[1].copy_(torch.linspace(1, -2, 10))
    sampler.ready.fill_(True)
    return sampler


class ImportanceLocalTimes:
    """The importance sampler's local times: its quantile map of uniform draws."""

    def __init__(self):
        self.sampler = PiecewiseImportanceTimeSampler(bins=8, uniform_mix=0.1)
        with torch.no_grad():
            self.sampler.clock_logits.copy_(torch.linspace(-2, 2, 8, dtype=torch.float64))

    def sample(self, count, device):
        return self.sampler(self.sampler.uniform.sample(count, device))

    def time_density(self, time):
        return self.sampler.time_density(time)


CONTINUOUS = {
    "uniform": lambda: UniformTimeSampler(1e-3),
    "stratified_uniform": StratifiedUniformTimeSampler,
    "piecewise": lambda: PiecewiseTimeSampler([1, 5, 2, 0.5]),
    "loss_derivative": fitted_loss_derivative,
    "loss_derivative_off": lambda: LossDerivativeTimeSampler(adaptive=False),
    "cdcd": fitted_cdcd_time_sampler,
    "cdcd_unfitted": lambda: CDCDTimeSampler(adaptive=True),
    "piecewise_importance": ImportanceLocalTimes,
}


@pytest.mark.parametrize("name", sorted(CONTINUOUS))
def test_density_integrates_to_one_and_matches_histogram(name):
    sampler = CONTINUOUS[name]()
    grid = (torch.arange(20000, dtype=torch.float64) + 0.5) / 20000
    density = sampler.time_density(grid.to(torch.float32)).double()
    assert torch.all(density >= 0)
    assert density.mean().item() == pytest.approx(1.0, abs=2e-3)
    torch.manual_seed(0)
    draws = torch.cat([sampler.sample(4096, "cpu").double() for _ in range(25)])
    bins = 20
    counts = torch.histc(draws, bins=bins, min=0, max=1)
    empirical = counts / len(draws) * bins
    expected = density.reshape(bins, -1).mean(-1)
    torch.testing.assert_close(empirical, expected, atol=0.04, rtol=0.04)


def test_discrete_density_is_step_probability_over_uniform_steps():
    base = PiecewiseTimeSampler([1, 3, 3, 1])
    sampler = DiscreteTimeSampler(base, steps=8)
    torch.manual_seed(0)
    draws = torch.cat([sampler.sample(4096, "cpu") for _ in range(25)])
    steps = torch.arange(1, 9) / 8
    probability = torch.stack([(draws == s).float().mean() for s in steps])
    torch.testing.assert_close(probability * 8, sampler.time_density(steps), atol=0.03, rtol=0)
    assert sampler.time_density(steps).mean().item() == pytest.approx(1.0)
    uniform = DiscreteTimeSampler(UniformTimeSampler(), steps=8)
    assert torch.equal(uniform.time_density(steps), torch.ones(8))


def test_samplers_without_a_density_say_so():
    assert FixedTimeSampler.time_density is None
    assert training_time_density(FixedTimeSampler(0.5), torch.ones(3)) is None
    with pytest.raises(ValueError, match="reports no draw density"):
        weight_objective(torch.ones(3, 2), _Corrected(FixedTimeSampler(0.5)), None)
    # Allocation: diffusion rows are uniform; reconstruction rows are a point mass.
    allocation = AdaptiveTimeAllocation()
    draw = allocation.sample(8, "cpu")
    density = allocation.time_density(draw.time)
    assert torch.isnan(density[draw.reconstruction]).all()
    assert torch.equal(density[~draw.reconstruction], torch.ones(int((~draw.reconstruction).sum())))

    class Undeclared:
        def sample(self, count, device, **context):
            return torch.rand(count)

    with pytest.raises(TypeError, match="does not declare time_density"):
        training_time_density(Undeclared(), torch.ones(2))


class _Corrected:
    objective_weighting = "importance_corrected"

    def __init__(self, base):
        self.time_density = base.time_density


# --- (a)/(b) adaptive noise schedules fit the true per-t loss ---------------


def kinked_loss(p):
    # Non-monotone, so the monotone fit is a projection whose answer depends on
    # the measure the observations represent; flat near 0 so both samplers'
    # epsilon floors see the same clean-end value.
    p = (p - 0.01).clamp_min(0)
    return p + 0.2 * torch.sin(3 * math.pi * p)


def smooth_loss(p):
    return p + 0.2 * torch.sin(3 * math.pi * p)


def fit(schedule_type, sampler, loss, use_density, **options):
    torch.manual_seed(0)
    schedule = schedule_type(
        LogLinear(epsilon=0), buffer_size=N, refit_every=1, warmup_steps=0, ema=0, **options
    )
    schedule.train()
    time = sampler.sample(N, "cpu").double()
    density = sampler.time_density(time) if use_density else None
    schedule.observe(time, loss(time), 0, training=True, density=density)
    assert schedule.has_schedule
    return schedule


def distance(a, b):
    return (a.alpha_vals - b.alpha_vals).abs().mean().item()


@pytest.mark.parametrize("weights", [[1, 8, 8, 1], [8, 1, 1, 8], [1, 2, 4, 8]])
def test_cdcd_schedule_fit_does_not_depend_on_the_training_time_sampler(weights):
    uniform = fit(CDCDAdaptiveSchedule, UniformTimeSampler(1e-3), kinked_loss, False)
    skewed = PiecewiseTimeSampler(weights)
    corrected = fit(CDCDAdaptiveSchedule, skewed, kinked_loss, True)
    uncorrected = fit(CDCDAdaptiveSchedule, skewed, kinked_loss, False)
    assert distance(corrected, uniform) < 5e-5
    assert distance(uncorrected, uniform) > 2e-3
    torch.testing.assert_close(
        corrected.loss_profile, uniform.loss_profile, atol=5e-4, rtol=0
    )


@pytest.mark.parametrize("coordinate", ["legacy_time", "physical"])
def test_sflm_schedule_fit_removes_the_training_time_sampler(coordinate):
    uniform = fit(SFLMAdaptiveSchedule, UniformTimeSampler(1e-3), smooth_loss, False, coordinate=coordinate)
    skewed = PiecewiseTimeSampler([1, 8, 8, 1])
    corrected = fit(SFLMAdaptiveSchedule, skewed, smooth_loss, True, coordinate=coordinate)
    uncorrected = fit(SFLMAdaptiveSchedule, skewed, smooth_loss, False, coordinate=coordinate)
    assert distance(corrected, uniform) < 1e-4
    assert 4 * distance(corrected, uniform) < distance(uncorrected, uniform)


@pytest.mark.parametrize("schedule_type", [CDCDAdaptiveSchedule, SFLMAdaptiveSchedule])
def test_uniform_density_reproduces_the_unweighted_fit_exactly(schedule_type):
    sampler = UniformTimeSampler(1e-3)
    old = fit(schedule_type, sampler, kinked_loss, False)
    new = fit(schedule_type, sampler, kinked_loss, True)
    for name, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[name]), name


def test_uniform_draws_make_the_unchanged_observe_call():
    calls = []

    class Recording(torch.nn.Module):
        loss_signal = "objective"

        def observe(self, time, loss, step, *, training, **options):
            calls.append(options)

    prediction = torch.zeros(3, 2, 4).log_softmax(-1)
    tokens = torch.zeros(3, 2, dtype=torch.long)
    per_token, mask = torch.rand(3, 2), torch.ones(3, 2)
    for density, expected in [(None, {}), (torch.ones(3), {})]:
        observe_training_loss(
            Recording(), torch.rand(3), prediction, tokens, per_token, mask,
            step=0, training=True, density=density,
        )
        assert calls.pop() == expected
    density = torch.tensor([0.5, 1.0, 2.0])
    observe_training_loss(
        Recording(), torch.rand(3), prediction, tokens, per_token, mask,
        step=0, training=True, density=density,
    )
    assert torch.equal(calls.pop()["density"], density)


def test_gumbel_fit_weights_squared_errors_by_inverse_density():
    def statistics(density):
        schedule = GumbelSchedule(learnable=True)
        schedule.train()
        time = torch.tensor([0.2, 0.5, 0.8])
        schedule.observe(time, torch.tensor([1.0, 2.0, 3.0]), 0, training=True, density=density)
        return schedule.pending.clone()

    assert torch.equal(statistics(None), statistics(torch.ones(3)))
    # Halving the density of one draw counts it twice: like a duplicated draw.
    weighted = statistics(torch.tensor([0.5, 1.0, 1.0]))
    schedule = GumbelSchedule(learnable=True)
    schedule.train()
    time = torch.tensor([0.2, 0.2, 0.5, 0.8])
    schedule.observe(time, torch.tensor([1.0, 1.0, 2.0, 3.0]), 0, training=True)
    torch.testing.assert_close(weighted[:-1], schedule.pending[:-1])


# --- (c) objective weighting -------------------------------------------------


def test_importance_corrected_keeps_the_expected_objective():
    def per_t_loss(t):
        return torch.exp(-3 * t) + t

    exact = (1 - math.exp(-3)) / 3 + 0.5
    # Plain (not antithetic) Monte Carlo draws; a negligible epsilon floor.
    options = dict(epsilon=1e-6, antithetic=False)
    sampler = PiecewiseTimeSampler(
        [1, 8, 8, 1], objective_weighting="importance_corrected", **options
    )
    torch.manual_seed(0)
    time = sampler.sample(400000, "cpu").double()
    density = sampler.time_density(time)
    per_token = per_t_loss(time)[:, None].expand(-1, 3)
    corrected = weight_objective(per_token, sampler, density).mean().item()
    assert corrected == pytest.approx(exact, abs=3e-3)
    uncorrected_sampler = PiecewiseTimeSampler(
        [1, 8, 8, 1], objective_weighting="uncorrected", **options
    )
    uncorrected = weight_objective(per_token, uncorrected_sampler, density).mean().item()
    # Oversampling re-weights the objective by the density: E_q[f] = int f q.
    grid = (torch.arange(100000, dtype=torch.float64) + 0.5) / 100000
    reweighted = (per_t_loss(grid) * sampler.time_density(grid)).mean().item()
    assert uncorrected == pytest.approx(reweighted, abs=2e-3)
    assert abs(uncorrected - exact) > 0.02


def tiny(cfg):
    cfg.model.hidden_size = 32
    cfg.model.cond_dim = 16
    cfg.model.n_heads = 4
    cfg.model.n_blocks = 1
    cfg.model.dropout = 0
    cfg.task.length = 6
    cfg.noise_schedule.warmup_steps = 0
    cfg.noise_schedule.buffer_size = 8
    cfg.noise_schedule.refit_every = 1
    return cfg


class TimeObjective(torch.nn.Module):
    """A per-token loss that depends only on the row's training time."""

    requires_context = True

    def forward(self, prediction, tokens, *, context):
        return (1 + context.time)[:, None].expand(tokens.shape) + 0 * prediction[..., 0]


@pytest.mark.parametrize("weighting", ["importance_corrected", "uncorrected"])
def test_recipe_loss_follows_the_declared_weighting(weighting):
    cfg = tiny(compose("cdcd_mid_range_times", [f"time_sampler.objective_weighting={weighting}"]))
    algo = instantiate(cfg.algorithm)
    algo.objective = TimeObjective()
    seen = {}
    sample = algo.time_sampler.sample

    def recording(*args, **kwargs):
        seen["time"] = sample(*args, **kwargs)
        return seen["time"]

    algo.time_sampler.sample = recording
    tokens = torch.randint(0, cfg.task.vocab_size, (16, cfg.task.length))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    loss = algo.loss(tokens, mask, step=0, training=True)["loss"]
    time = seen["time"]
    # The objective reads the schedule's training-time coordinate.
    local = algo.views[0].schedule.training_time(time)
    density = algo.time_sampler.time_density(time)
    expected = (1 + local) / density if weighting == "importance_corrected" else 1 + local
    torch.testing.assert_close(loss, expected.mean())


def test_recipe_adaptive_schedule_receives_the_draw_density():
    cfg = tiny(compose("cdcd_mid_range_times"))
    algo = instantiate(cfg.algorithm)
    clock = algo.views[0].schedule.clock
    tokens = torch.randint(0, cfg.task.vocab_size, (8, cfg.task.length))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    algo.loss(tokens, mask, step=0, training=True)["loss"].backward()
    weights = clock.draw_weight_buf
    assert not torch.all(weights == 1)
    # Each buffered weight is 1 / density of the bin its draw fell in.
    candidates = 1 / (4 * algo.time_sampler.probabilities)
    assert torch.isclose(weights[:, None], candidates, rtol=1e-6).any(-1).all()
    assert clock.has_schedule
    # Validation feeds nothing back.
    before = clock.draw_weight_buf.clone()
    algo.loss(tokens, mask, step=1, training=False)
    assert torch.equal(before, clock.draw_weight_buf)


def test_uniform_weightings_give_the_same_loss_and_gradients():
    results = []
    for weighting in ("importance_corrected", "uncorrected"):
        cfg = tiny(compose("cdcd", [f"+time_sampler.objective_weighting={weighting}"]))
        torch.manual_seed(0)
        algo = instantiate(cfg.algorithm)
        tokens = torch.randint(0, cfg.task.vocab_size, (4, cfg.task.length))
        mask = torch.ones_like(tokens, dtype=torch.bool)
        torch.manual_seed(1)
        loss = algo.loss(tokens, mask, step=0, training=True)["loss"]
        loss.backward()
        grads = [p.grad.clone() for p in algo.parameters() if p.grad is not None]
        results.append((loss.detach(), grads, algo.views[0].schedule.clock.state_dict()))
    (loss_a, grads_a, state_a), (loss_b, grads_b, state_b) = results
    assert torch.equal(loss_a, loss_b)
    assert all(torch.equal(a, b) for a, b in zip(grads_a, grads_b))
    assert all(torch.equal(state_a[k], state_b[k]) for k in state_a)


def test_objective_that_divides_by_density_is_not_corrected_twice():
    class Divides:
        divides_by_time_density = True

    per_token = torch.rand(4, 3)
    sampler = PiecewiseTimeSampler([1, 3])
    density = torch.tensor([0.5, 1.5, 0.5, 1.5])
    assert weight_objective(per_token, sampler, density, Divides()) is per_token
    torch.testing.assert_close(
        weight_objective(per_token, sampler, density), per_token / density[:, None]
    )


# --- (e) generation never consults the training time sampler ----------------


class Untouchable:
    def __getattr__(self, name):
        raise AssertionError(f"generation consulted the training time sampler ({name})")


def test_generation_never_consults_the_training_time_sampler():
    cfg = tiny(compose("cdcd_mid_range_times"))
    cfg.sampler.steps = 3
    algo = instantiate(cfg.algorithm)
    tokens = torch.randint(0, cfg.task.vocab_size, (8, cfg.task.length))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    algo.loss(tokens, mask, step=0, training=True)["loss"].backward()
    assert algo.views[0].schedule.clock.has_schedule
    tokens, mask = tokens[:2], mask[:2]
    prompt = torch.zeros_like(mask)
    prompt[:, :2] = True
    algo.eval()
    torch.manual_seed(3)
    expected, _ = algo.generate(instantiate(cfg.sampler), tokens, prompt)
    algo.time_sampler = Untouchable()
    torch.manual_seed(3)
    result, _ = algo.generate(instantiate(cfg.sampler), tokens, prompt)
    assert torch.equal(result, expected)


# --- checkpoints and other assemblies ----------------------------------------


def test_schedule_checkpoint_without_draw_weights_loads_as_uniform():
    schedule = fit(CDCDAdaptiveSchedule, UniformTimeSampler(1e-3), kinked_loss, False)
    state = schedule.state_dict()
    del state["draw_weight_buf"]
    restored = CDCDAdaptiveSchedule(LogLinear(epsilon=0), buffer_size=N, ema=0)
    restored.load_state_dict(state)
    assert torch.equal(restored.draw_weight_buf, torch.ones(N, dtype=torch.float64))
    grid = torch.linspace(0, 1, 7, dtype=torch.float64)
    torch.testing.assert_close(restored(grid)[0], schedule(grid)[0])


@pytest.mark.parametrize("continuous", [False, True])
def test_variable_length_importance_correction_divides_every_loss_term(continuous):
    from core.tests.test_variable_length import assembly

    class MidTimes(PiecewiseTimeSampler):
        # Every draw in the second bin: density 2 * 0.75 = 1.5.
        def sample(self, count, device, **context):
            return torch.full((count,), 0.6, device=device)

    algo = assembly(continuous=continuous)
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    mask = torch.ones(2, 4)
    results = {}
    for weighting in ("uncorrected", "importance_corrected"):
        algo.algorithm.time_sampler = MidTimes([1, 3], objective_weighting=weighting)
        torch.manual_seed(5)
        results[weighting] = algo.loss(tokens, mask, training=True)
    for key in ("loss", "content_loss", "insertion_loss", "insertion_content_loss"):
        torch.testing.assert_close(
            results["importance_corrected"][key] * 1.5, results["uncorrected"][key]
        )

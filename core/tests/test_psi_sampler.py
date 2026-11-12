"""Ψ-sampler transitions on the composed loop, for every corruption family."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from core.modifiers.hybrid.views import ViewPoint
from core.processes.absorbing import AbsorbingProcess
from core.processes.uniform import UniformProcess
from core.samplers.composed import ComposedSampler
from core.samplers.forward_resampling import ForwardResamplingSampler
from core.samplers.hybrid import HybridSampler
from core.samplers.policies import TokenSelection
from core.samplers.psi import KappaSchedule, LoopTimeWarp, PsiTransition
from core.tests.networks import TINY32
from core.tests.test_block_diffusion import open_output
from core.tests.test_sampler_equivalence import build_algorithm


def mdlm():
    from core.sweeps.configuration import compose

    torch.manual_seed(5)
    algo = instantiate(compose(overrides=[*TINY32, "task.length=8"]).algorithm).eval()
    open_output(algo.model.model if hasattr(algo.model, "model") else algo.model)
    return algo


@pytest.fixture(scope="module")
def duo(tmp_path_factory):
    from core.tests.test_uniform_sedd_duo import build_integral_table, config
    from core.training.module import TrainingModule

    path = tmp_path_factory.mktemp("duo") / "v12.npz"
    np.savez(path, **build_integral_table(12, num_points=2001))
    torch.manual_seed(5)
    return TrainingModule(config("duo", str(path))).algorithm.eval()


# Single views of every corruption family, two hybrids, and the MDLM / DUO presets.
RECIPES = [
    ("gaussian", None), ("simplex", None), ("spherical", None), ("vmf", None),
    ("blurring", None), ("masked", None), ("uniform", None),
    ("gaussian", "masked"), ("spherical", "uniform"), "mdlm", "duo",
]
IDS = [r if isinstance(r, str) else "+".join(p for p in r if p) for r in RECIPES]


def recipe(name, duo):
    if name == "mdlm":
        return mdlm()
    if name == "duo":
        return duo
    return build_algorithm(name[0], name[1], "none")


def psi(steps=4, **options):
    return ComposedSampler(steps, tokens=TokenSelection("posterior"), transition=PsiTransition(**options),
                           final_denoise="greedy")


def prompts(algo):
    length = getattr(algo, "length", None) or 8
    tokens = torch.arange(2 * length).reshape(2, length) % 9
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[:, :2] = True
    return tokens, mask


def generate(algo, sampler, seed=17):
    tokens, mask = prompts(algo)
    torch.manual_seed(seed)
    return algo.generate(sampler, tokens, mask)


@pytest.mark.parametrize("name", RECIPES, ids=IDS)
def test_kappa_one_is_native_and_kappa_zero_is_forward_resampling(name, duo):
    algo = recipe(name, duo)
    native, native_info = generate(algo, HybridSampler(steps=4))
    for schedule in ("rescale", "cap", "constant"):
        output, info = generate(algo, psi(schedule=schedule, eta=0.0))
        assert torch.equal(output, native) and info == native_info
    renoised, renoised_info = generate(algo, ForwardResamplingSampler(steps=4))
    output, info = generate(algo, psi(schedule="constant", eta=1.0))
    assert torch.equal(output, renoised) and info == renoised_info


@pytest.mark.parametrize("name", RECIPES, ids=IDS)
@pytest.mark.parametrize("schedule", ["rescale", "cap", "loop", "constant"])
def test_prompts_fixed_and_nfe_unchanged(name, schedule, duo):
    algo = recipe(name, duo)
    tokens, mask = prompts(algo)
    sampler = psi(steps=5, schedule=schedule, eta=0.3, t_on=0.6, t_off=0.2)
    assert sampler.steps_for_nfe(64) == HybridSampler(steps=5).steps_for_nfe(64) == 63
    for seed in (0, 1):
        output, info = generate(algo, sampler, seed)
        assert torch.equal(output[mask], tokens[mask])
        assert info["nfe"] == 6 and info["nfe_per_sample"] == [6, 6]


def token_view(process):
    return SimpleNamespace(process=process, representation=SimpleNamespace(encode=lambda x: x),
                           update=lambda p, r, state, *args, **kw: state)


def point(keep):
    return ViewPoint(torch.zeros(1), (torch.tensor([keep], dtype=torch.float64),))


def call(transition, view, state, logits, keep_t, keep_s, generated):
    probabilities = logits.exp()
    source, destination = [point(keep_t)], [point(keep_s)]
    step = transition.prepare([view], logits, probabilities, source, destination, generated,
                              time=1 - torch.tensor([keep_t]), next_time=1 - torch.tensor([keep_s]))
    return transition(view, state, probabilities, None, source[0], destination[0], step=step,
                      final_step=False, generated=generated, prompt_tokens=state,
                      released=None, previous_tokens=state)


def test_revealed_tokens_remask_only_when_kappa_below_one():
    view = token_view(AbsorbingProcess(4))
    state = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]).repeat(64, 1)  # all revealed
    logits = torch.log_softmax(torch.randn(64, 8, 5), -1)
    generated = torch.ones_like(state, dtype=torch.bool)
    torch.manual_seed(0)
    assert torch.equal(call(PsiTransition(eta=0.0), view, state, logits, 0.5, 0.6, generated), state)
    remasked = call(PsiTransition("constant", eta=0.5), view, state, logits, 0.5, 0.6, generated)
    assert (remasked == 4).any()  # re-noised to α_s = 0.6: masks reappear


def test_uniform_forward_branch_has_full_support():
    view = token_view(UniformProcess(6))
    state = torch.zeros(256, 4, dtype=torch.long)
    logits = torch.log(torch.nn.functional.one_hot(torch.zeros_like(state), 6).float() + 1e-12)
    torch.manual_seed(0)
    draws = call(PsiTransition("constant", eta=1.0), view, state, logits, 0.3, 0.4,
                 torch.ones_like(state, dtype=torch.bool))
    assert set(draws.unique().tolist()) == set(range(6))  # a certain x_θ, yet every token


def test_kappa_schedules_follow_remdm_sigmas():
    keep_t, keep_s = torch.tensor([0.0, 0.2, 0.9]), torch.tensor([0.1, 0.3, 0.95])
    sigma_max = torch.tensor([1.0, 0.7 / 0.2, 0.05 / 0.9]).clamp(max=1)
    room = 1 - keep_s
    for schedule, sigma in [("rescale", 0.02 * sigma_max), ("cap", sigma_max.clamp(max=0.02))]:
        kappa = KappaSchedule(schedule, 0.02)(keep_t, keep_s)
        torch.testing.assert_close(kappa, (1 - sigma / room).double())
    assert (KappaSchedule("rescale", 1.0)(keep_t, keep_s) >= 0).all()
    assert (KappaSchedule("constant", 0.05)(keep_t, keep_s) == 0.95).all()
    loop = KappaSchedule("loop", 0.02, alpha_on=0.9)
    hold = loop(torch.tensor([0.9, 0.2]), torch.tensor([0.9, 0.3]))
    assert hold[0].item() == pytest.approx(1 - 0.02 / 0.1) and hold[1].item() == 1


def test_kappa_is_the_same_function_of_grid_time_for_every_corruption():
    # κ reads α = 1 - t on the sampler's grid, so one η and schedule give one κ curve
    # whatever the view's own schedule (Gaussian SNR, simplex mixing, discrete keep rate).
    time, next_time = torch.tensor([0.7, 0.3]), torch.tensor([0.6, 0.2])
    kappas = []
    for family in ("gaussian", "simplex", "spherical", "vmf", "masked", "uniform"):
        algo = build_algorithm(family, None, "none")
        source = algo.points(time, corruption_mask=None)
        destination = algo.points(next_time, corruption_mask=None)
        logits = torch.zeros(2, 4, 12).log_softmax(-1)
        generated = torch.ones(2, 4, dtype=torch.bool)
        step = PsiTransition("rescale", eta=0.5).prepare(
            algo.views, logits, logits.exp(), source, destination, generated,
            time=time, next_time=next_time)
        kappas.append(step.kappa)
    expected = KappaSchedule("rescale", 0.5)(1 - time.double(), 1 - next_time.double())
    for kappa in kappas:
        torch.testing.assert_close(kappa.double(), expected)


def test_loop_warp_is_one_minus_the_target_profile():
    algo = build_algorithm("gaussian", None, "none")
    times = torch.linspace(1, 1e-5, 41, dtype=torch.float64)
    warp = LoopTimeWarp(t_on=0.6, t_off=0.1, alpha_on=0.9)
    warped = warp(algo, times)
    alpha = 1 - warped
    plateau = (times <= 0.6) & (times >= 0.1)
    assert len(set(warped[plateau].tolist())) == 1
    torch.testing.assert_close(alpha[plateau], torch.full_like(alpha[plateau], 0.9))
    assert (alpha[1:] >= alpha[:-1] - 1e-12).all()  # monotone: cleaner as time runs down
    assert warped.min() >= times.min() and warped.max() <= times.max()
    other = build_algorithm("masked", None, "none")
    assert torch.equal(warp(other, times), warped)  # the same grid for every recipe


def test_refuses_commitments():
    with pytest.raises(ValueError, match="commitment=none"):
        generate(mdlm(), ComposedSampler(3, tokens=TokenSelection("sample"), transition=PsiTransition(),
                                         commitment="freeze"))


@pytest.mark.parametrize("config", ["psi", "uniform_psi"])
def test_sampler_configs_instantiate(config):
    sampler = instantiate(OmegaConf.load(f"configs/inference/sampler/{config}.yaml"))
    assert isinstance(sampler, ComposedSampler) and isinstance(sampler.transition, PsiTransition)
    assert sampler.transition.kappa.schedule == "rescale" and sampler.transition.kappa.eta == 0.02


@pytest.mark.parametrize("name", [("gaussian", None), ("gaussian", "masked")], ids=["gaussian", "gaussian+masked"])
def test_log_snr_clock_reads_the_gaussian_views_signal_to_noise(name, duo):
    # clock=log_snr: α = sigmoid(log-SNR) of the Gaussian view at its own time, also in a
    # hybrid whose discrete view would otherwise set the clock.
    algo = recipe(name, duo)
    index = next(i for i, v in enumerate(algo.views) if type(v.process).__name__ == "GaussianProcess")
    time, next_time = torch.tensor([0.7, 0.3]), torch.tensor([0.6, 0.2])
    source, destination = algo.points(time), algo.points(next_time)
    logits = torch.zeros(2, 4, 12).log_softmax(-1)
    generated = torch.ones(2, 4, dtype=torch.bool)
    transition = PsiTransition("rescale", eta=0.5, clock="log_snr")
    transition.validate(algo.views, "none")
    step = transition.prepare(algo.views, logits, logits.exp(), source, destination, generated,
                              time=time, next_time=next_time)

    def sigmoid_log_snr(point):
        c = point.coefficients
        return torch.sigmoid(torch.log(c.alpha.double() ** 2 / c.sigma.double() ** 2))

    expected = KappaSchedule("rescale", 0.5)(sigmoid_log_snr(source[index]), sigmoid_log_snr(destination[index]))
    torch.testing.assert_close(step.kappa.double(), expected)
    controller = PsiTransition("rescale", eta=0.5).prepare(
        algo.views, logits, logits.exp(), source, destination, generated, time=time, next_time=next_time)
    assert not torch.allclose(controller.kappa.double(), expected)  # the two clocks differ


def test_log_snr_loop_warp_holds_the_gaussian_alpha_on_its_plateau():
    algo = build_algorithm("gaussian", None, "none")
    times = torch.linspace(1, 1e-5, 41, dtype=torch.float64)
    warped = LoopTimeWarp(t_on=0.6, t_off=0.1, alpha_on=0.9, clock="log_snr")(algo, times)
    c = algo.points(warped)[0].coefficients
    alpha = (c.alpha.double() ** 2 / (c.alpha.double() ** 2 + c.sigma.double() ** 2))
    plateau = (times <= 0.6) & (times >= 0.1)
    torch.testing.assert_close(alpha[plateau], torch.full_like(alpha[plateau], 0.9), atol=1e-5, rtol=0)
    assert len(set(warped[plateau].tolist())) == 1


def test_log_snr_clock_refuses_recipes_without_a_gaussian_view():
    with pytest.raises(TypeError, match="Gaussian view"):
        PsiTransition(clock="log_snr").validate(mdlm().views, "none")
    with pytest.raises(ValueError, match="clock"):
        PsiTransition(clock="time")

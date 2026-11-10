"""Ψ-sampler transitions on the composed loop (masked, uniform and hybrid recipes)."""

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from core.modifiers.hybrid.views import ViewPoint
from core.processes.absorbing import AbsorbingProcess
from core.processes.uniform import UniformProcess
from core.samplers.composed import ComposedSampler
from core.samplers.policies import ReverseTransition, TokenSelection
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


def recipe(name, duo):
    if name == "mdlm":
        return mdlm()
    if name == "duo":
        return duo
    return build_algorithm("gaussian", name, "none")  # a hybrid with a masked/uniform view


def composed(transition, steps=4, **options):
    defaults = dict(tokens=TokenSelection("posterior"), final_denoise="greedy", precision="float64")
    return ComposedSampler(steps, transition=transition, **{**defaults, **options})


def prompts(vocab=9, length=8):
    tokens = torch.arange(2 * length).reshape(2, length) % vocab
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[:, :2] = True
    return tokens, mask


def generate(algo, sampler, seed=17):
    tokens, mask = prompts()
    torch.manual_seed(seed)
    return algo.generate(sampler, tokens, mask)


RECIPES = ["mdlm", "duo", "masked", "uniform"]


@pytest.mark.parametrize("name", RECIPES)
def test_kappa_one_is_the_ancestral_sampler(name, duo):
    algo = recipe(name, duo)
    native, native_info = generate(algo, composed(ReverseTransition()))
    for schedule in ("rescale", "cap"):
        psi, info = generate(algo, composed(PsiTransition(schedule, eta=0.0)))
        assert torch.equal(psi, native)
        assert info == native_info


@pytest.mark.parametrize("name", RECIPES)
@pytest.mark.parametrize("schedule", ["rescale", "cap", "loop"])
def test_prompts_fixed_and_nfe_unchanged(name, schedule, duo):
    algo = recipe(name, duo)
    tokens, mask = prompts()
    reference = composed(ReverseTransition(), steps=5)
    sampler = composed(PsiTransition(schedule, eta=0.08, t_on=0.6, t_off=0.2), steps=5)
    assert sampler.steps_for_nfe(64) == reference.steps_for_nfe(64) == 63
    for seed in (0, 1, 2):
        output, info = generate(algo, sampler, seed)
        assert torch.equal(output[mask], tokens[mask])
        assert info["nfe"] == 6 and info["nfe_per_sample"] == [6, 6]


def absorbing_view(mask_index=4):
    from types import SimpleNamespace

    return SimpleNamespace(process=AbsorbingProcess(mask_index), representation=None, update=None)


def point(keep):
    return ViewPoint(torch.zeros(1), (torch.tensor([keep], dtype=torch.float64),))


def test_unmasked_tokens_remask_only_when_kappa_below_one():
    view = absorbing_view()
    state = torch.tensor([[0, 1, 2, 3, 4, 4]])
    x = torch.softmax(torch.randn(1, 6, 5, dtype=torch.float64), -1)
    source, destination = point(0.5), point(0.6)
    exact = PsiTransition("rescale", eta=0.0)
    assert exact.probabilities(view, state, x, source, destination) is None
    psi = PsiTransition("rescale", eta=0.5)
    mixture = psi.probabilities(view, state, x, source, destination)
    unmasked = state.ne(4)
    assert (mixture[unmasked][:, 4] > 0).all()  # remasking mass on revealed tokens
    torch.testing.assert_close(mixture.sum(-1), torch.ones(1, 6, dtype=torch.float64))
    # Paper A.4: revealed z = x gets [κ + (1-κ)α_s] x + (1-κ)(1-α_s) m.
    kappa = KappaSchedule("rescale", 0.5)(torch.tensor([0.5]), torch.tensor([0.6])).item()
    assert mixture[0, 0, 0].item() == pytest.approx(kappa + (1 - kappa) * 0.6)
    torch.manual_seed(0)
    draws = torch.stack([psi(view, state, x, None, source, destination) for _ in range(200)])
    assert (draws[:, unmasked] == 4).any()
    torch.manual_seed(0)
    native = ReverseTransition()
    view.update = lambda process, representation, state, p, s, d, noise=None: process.reverse(
        state, p, s.coefficients[0], d.coefficients[0]
    )
    kept = torch.stack([native(view, state, x, None, source, destination, final_step=False,
                               generated=None, prompt_tokens=None, released=None,
                               previous_tokens=None) for _ in range(200)])
    assert (kept[:, unmasked] == state[unmasked]).all()


def test_uniform_mixture_keeps_full_support():
    from types import SimpleNamespace

    view = SimpleNamespace(process=UniformProcess(6), representation=None, update=None)
    state = torch.tensor([[0, 1, 2]])
    x = torch.nn.functional.one_hot(torch.tensor([[0, 0, 0]]), 6).double()  # a certain x_θ
    mixture = PsiTransition("cap", eta=0.05).probabilities(view, state, x, point(0.3), point(0.5))
    assert (mixture > 0).all()
    torch.testing.assert_close(mixture.sum(-1), torch.ones(1, 3, dtype=torch.float64))


def test_kappa_schedules_follow_remdm_sigmas():
    keep_t, keep_s = torch.tensor([0.0, 0.2, 0.9]), torch.tensor([0.1, 0.3, 0.95])
    sigma_max = torch.tensor([1.0, 0.7 / 0.2, 0.05 / 0.9]).clamp(max=1)
    room = 1 - keep_s
    for schedule, sigma in [
        ("rescale", 0.02 * sigma_max),
        ("cap", sigma_max.clamp(max=0.02)),
    ]:
        kappa = KappaSchedule(schedule, 0.02)(keep_t, keep_s)
        torch.testing.assert_close(kappa, (1 - sigma / room).double())
    assert (KappaSchedule("rescale", 1.0)(keep_t, keep_s) >= 0).all()
    loop = KappaSchedule("loop", 0.02, alpha_on=0.9)
    hold = loop(torch.tensor([0.9, 0.2]), torch.tensor([0.9, 0.3]))
    assert hold[0].item() == pytest.approx(1 - 0.02 / 0.1) and hold[1].item() == 1


def test_loop_warp_holds_alpha_on_its_plateau():
    algo = mdlm()
    times = torch.linspace(1, 1e-5, 41, dtype=torch.float64)
    warped = LoopTimeWarp(t_on=0.6, t_off=0.1, alpha_on=0.9)(algo, times)
    keep = algo.points(warped)[0].coefficients[0].double()
    plateau = (times <= 0.6) & (times >= 0.1)
    torch.testing.assert_close(keep[plateau], torch.full_like(keep[plateau], 0.9), atol=1e-6, rtol=0)
    assert len(set(warped[plateau].tolist())) == 1
    assert (keep[1:] >= keep[:-1] - 1e-9).all()  # still a forward-in-keep grid
    above = times > 0.6
    # Linear ramp to α_on, floored at the schedule's own keep rate at t = 1.
    ramp = ((1 - times[above]) * 0.9 / 0.4).clamp(min=keep[0].item())
    torch.testing.assert_close(keep[above], ramp, atol=1e-6, rtol=0)


def test_refuses_recipes_without_a_discrete_view_and_commitments():
    algo = build_algorithm("gaussian", None, "none")
    with pytest.raises(TypeError, match="discrete view"):
        generate(algo, composed(PsiTransition()))
    with pytest.raises(ValueError, match="commitment=none"):
        generate(mdlm(), composed(PsiTransition(), commitment="freeze", tokens=TokenSelection("sample")))


@pytest.mark.parametrize("config", ["psi", "uniform_psi"])
def test_sampler_configs_instantiate(config):
    sampler = instantiate(OmegaConf.load(f"configs/inference/sampler/{config}.yaml"))
    assert isinstance(sampler, ComposedSampler) and isinstance(sampler.transition, PsiTransition)
    assert sampler.transition.kappa.schedule == "rescale" and sampler.transition.kappa.eta == 0.02

import pytest
import torch
from scipy.special import ive
from core.processes.vmf import VMFProcess
from core.noise_schedules.vmf import VMFSchedule
from core.tests.test_sampler_equivalence import build_algorithm
from core.samplers.composed import ComposedSampler
from core.samplers.correctors import SphericalLangevinCorrector


@pytest.mark.parametrize(
    "dimension,kappa", [(3, 0.0), (3, 5.0), (32, 15.0), (512, 250.0)]
)
def test_forward_moments(dimension, kappa):
    torch.manual_seed(12)
    process = VMFProcess(dimension, max(kappa, 1.0))
    clean = torch.zeros(1, 12000, dimension)
    clean[..., 0] = 1
    sample = process.corrupt_state(
        clean, (torch.tensor([kappa]),), torch.ones(1, 12000).bool()
    )
    expected = (
        0 if kappa == 0 else ive(dimension / 2, kappa) / ive(dimension / 2 - 1, kappa)
    )
    assert abs(sample[..., 0].mean().item() - expected) < 0.015
    torch.testing.assert_close(
        sample.norm(dim=-1), torch.ones(1, 12000), atol=2e-6, rtol=0
    )


def test_flux_equation_and_endpoints():
    process = VMFProcess(32, 15.0)
    s = torch.linspace(-0.8, 0.8, 1001)
    k = torch.tensor(10.0)
    psi = process._lookup((s + 1) / 2, k, 0)
    derivative = torch.gradient(psi, spacing=(s,))[0]
    a = ive(16, 10) / ive(15, 10)
    residual = (1 - s * s) * derivative + (10 * (1 - s * s) - 31 * s) * psi - (a - s)
    assert residual[2:-2].abs().max() < 0.015
    torch.testing.assert_close(
        process._lookup((s + 1) / 2, torch.tensor(0.0), 0), torch.full_like(s, 1 / 31)
    )


@pytest.mark.parametrize("correct", [False, True])
def test_model_training_and_sampling(correct):
    algo = build_algorithm("vmf", None, "none")
    tokens = torch.randint(0, 12, (2, 8))
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[:, :2] = True
    loss = algo.loss(tokens, ~mask, training=True)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in algo.parameters() if p.grad is not None
    )
    sampler = ComposedSampler(
        steps=2, corrector=SphericalLangevinCorrector() if correct else None
    )
    output, info = sampler.sample(algo, tokens, mask)
    assert info["nfe"] == (5 if correct else 3)
    assert torch.equal(output[:, :2], tokens[:, :2])
    assert output.shape == tokens.shape


def test_schedule_direction():
    k, derivative = VMFSchedule(15)(torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(k, torch.tensor([15.0, 0.0]))
    assert (derivative == -15).all()


@pytest.mark.parametrize("sc", ["hidden", "posterior_mean"])
def test_forward_resampling_and_hybrid(sc):
    from core.samplers.forward_resampling import ForwardResamplingSampler

    algo = build_algorithm("vmf", "uniform", sc)
    tokens = torch.randint(0, 12, (2, 8))
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    loss = algo.loss(tokens, ~mask, training=True)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    output, info = ForwardResamplingSampler(steps=2).sample(algo, tokens, mask)
    assert output.shape == tokens.shape
    assert info["nfe"] == 3


def test_conditional_flow_transports_uniform_to_vmf():
    from core.processes.spherical import sphere_normalize

    torch.manual_seed(8)
    process = VMFProcess(3, 5.0)
    x = sphere_normalize(torch.randn(1, 6000, 3))
    codebook = torch.tensor([[1.0, 0.0, 0.0]])
    probabilities = torch.ones(1, 6000, 1)
    for kappa in torch.linspace(0, 5, 401)[:-1]:
        velocity = process.velocity(x, codebook, probabilities, kappa)
        x = sphere_normalize(x + (5 / 400) * velocity)
    expected = ive(1.5, 5) / ive(0.5, 5)
    assert abs(x[..., 0].mean().item() - expected) < 0.015

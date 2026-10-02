"""SNR-equivalent paths, derivative transport, and adaptive composition."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from core.noise_schedules.adaptive import CDCDGaussianSchedule, SFLMGaussianSchedule
from core.noise_schedules.embedding_rank import EmbeddingSphereRankSchedule
from core.noise_schedules.exponential_ve import ExponentialVESchedule
from core.noise_schedules.gaussian_path import GaussianPath
from core.noise_schedules.gaussian_rank import OneHotRankSchedule
from core.noise_schedules.linear_interpolation import LinearInterpolationSchedule
from core.noise_schedules.rescaled_gaussian import RescaledGaussianSchedule
from core.representations.spec import RepresentationSpec


@pytest.mark.parametrize("kind", ["ve", "vp", "linear"])
@pytest.mark.parametrize(
    "base", [ExponentialVESchedule(), LinearInterpolationSchedule()]
)
def test_snr_argmax_and_derivatives(kind, base):
    time = torch.linspace(0.03, 0.97, 23, dtype=torch.float64)
    schedule = RescaledGaussianSchedule(base, kind)
    before, after = base(time), schedule(time)
    torch.testing.assert_close(before.sigma / before.alpha, after.sigma / after.alpha)
    if kind == "ve":
        torch.testing.assert_close(after.alpha, torch.ones_like(time))
    elif kind == "vp":
        torch.testing.assert_close(
            after.alpha.square() + after.sigma.square(), torch.ones_like(time)
        )
    else:
        torch.testing.assert_close(after.alpha + after.sigma, torch.ones_like(time))
    noise = torch.randn(
        23, 12, generator=torch.Generator().manual_seed(7), dtype=time.dtype
    )
    clean = torch.eye(12, dtype=time.dtype)[torch.arange(23) % 12]
    x = before.alpha[:, None] * clean + before.sigma[:, None] * noise
    y = after.alpha[:, None] * clean + after.sigma[:, None] * noise
    assert torch.equal(x.argmax(-1), y.argmax(-1))
    epsilon = 1e-6
    left, right = schedule(time - epsilon), schedule(time + epsilon)
    torch.testing.assert_close(
        after.dalpha_dt, (right.alpha - left.alpha) / (2 * epsilon)
    )
    torch.testing.assert_close(
        after.dsigma_dt, (right.sigma - left.sigma) / (2 * epsilon)
    )
    assert not hasattr(schedule, "observe")


def test_endpoints_and_invalid_paths():
    base = LinearInterpolationSchedule()
    for kind in ("vp", "linear"):
        path = RescaledGaussianSchedule(base, kind)(torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(path.alpha, torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(path.sigma, torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match="singular"):
        RescaledGaussianSchedule(base, "ve")(torch.ones(1))
    with pytest.raises(ValueError, match="path_type"):
        RescaledGaussianSchedule(base, "other")

    class Invalid(LinearInterpolationSchedule):
        def forward(self, t):
            return GaussianPath(
                t, torch.zeros_like(t), torch.ones_like(t), torch.zeros_like(t)
            )

    for kind in ("ve", "vp", "linear"):
        for t in (0.0, -1.0, float("nan")):
            with pytest.raises(ValueError):
                RescaledGaussianSchedule(Invalid(), kind)(torch.tensor([t]))


@pytest.mark.parametrize("cls", [SFLMGaussianSchedule, CDCDGaussianSchedule])
@pytest.mark.parametrize("outer_adaptive", [False, True])
def test_adaptive_feedback_checkpoint_and_derivatives(cls, outer_adaptive):
    def make():
        base = ExponentialVESchedule(training_epsilon=0.03)
        options = dict(
            buffer_size=32,
            refit_every=1,
            warmup_steps=0,
            n_grid=101,
            coordinate="physical",
        )
        if outer_adaptive:
            return cls(RescaledGaussianSchedule(base), **options)
        return RescaledGaussianSchedule(cls(base, **options))

    schedule = make()
    time = torch.linspace(0.01, 0.99, 32, dtype=torch.float64)
    torch.testing.assert_close(schedule.training_time(time), time * 0.97)
    schedule.observe(time, time.square() + 0.1, 0, training=True)
    restored = make()
    restored.load_state_dict(schedule.state_dict())
    for name in ("alpha", "sigma", "dalpha_dt", "dsigma_dt"):
        torch.testing.assert_close(
            getattr(schedule(time), name), getattr(restored(time), name)
        )
    assert schedule.snapshot() == restored.snapshot()
    path = schedule(time + 0.000123)
    left, right = schedule(time + 0.000123 - 1e-6), schedule(time + 0.000123 + 1e-6)
    torch.testing.assert_close(
        path.dalpha_dt, (right.alpha - left.alpha) / 2e-6, atol=1e-7, rtol=1e-5
    )
    torch.testing.assert_close(
        path.dsigma_dt, (right.sigma - left.sigma) / 2e-6, atol=1e-7, rtol=1e-5
    )
    schedule.eval()
    state = {
        k: v.clone() if torch.is_tensor(v) else v
        for k, v in schedule.state_dict().items()
    }
    schedule.observe(time, time, 1, training=True)
    for key, value in state.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, schedule.state_dict()[key])


def test_representation_requirements_and_hydra_base():
    base = EmbeddingSphereRankSchedule(8, grid_size=65, quadrature_order=16)
    schedule = RescaledGaussianSchedule(base)
    assert schedule.representation_requirements == base.representation_requirements
    schedule.representation_requirements.validate(
        RepresentationSpec("embedding", 8, 12, True), "test"
    )
    incompatible = RescaledGaussianSchedule(OneHotRankSchedule(12, num_points=32))
    with pytest.raises(ValueError, match="incompatible"):
        incompatible.representation_requirements.validate(
            RepresentationSpec("embedding", 8, 12, True), "test"
        )
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[1] / "new_configs")
    ):
        config = compose(
            config_name="config",
            overrides=[
                "noise_schedule=rescaled_gaussian",
                "noise_schedule@noise_schedule.base=exponential_ve",
                "noise_schedule.path_type=linear",
            ],
        )
    schedule = instantiate(config.noise_schedule)
    assert isinstance(schedule.base, ExponentialVESchedule)
    assert schedule.path_type == "linear"

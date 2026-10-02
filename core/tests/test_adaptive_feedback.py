"""Training orchestration feeds schedules without changing the optimized loss."""

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from torch import nn

from core.algorithms.adaptive import observe_training_loss, validate_adaptive_feedback
from core.noise_schedules.duo_calibration import build_integral_table
from core.parameterizations.score import AbsorbingScoreParameterization
from core.tests.test_uniform_sedd_duo import config
from core.tests.test_mdlm import config as mdlm_config


class RecordingSchedule(nn.Module):
    def __init__(self, base=None, loss_signal="objective"):
        super().__init__()
        self.base = base
        self.loss_signal = loss_signal
        self.observations = []

    def forward(self, time):
        return self.base(time)

    def training_time(self, time):
        return self.base.training_time(time)

    def observe(self, time, loss, step, *, training):
        assert training
        assert not time.requires_grad and not loss.requires_grad
        self.observations.append((time.clone(), loss.clone(), step))


def test_masked_feedback_selects_signal_and_omits_empty_examples():
    prediction = torch.tensor(
        [[[0.2, 0.8], [0.7, 0.3]], [[0.4, 0.6], [0.1, 0.9]]],
        requires_grad=True,
    ).log()
    tokens = torch.tensor([[1, 0], [0, 1]])
    mask = torch.tensor([[0, 1], [0, 0]])
    objective = torch.tensor([[float("inf"), 7.0], [5.0, 6.0]], requires_grad=True)
    time = torch.tensor([0.25, 0.75], requires_grad=True)
    for signal, expected in [("objective", 7.0), ("cross_entropy", -np.log(0.7))]:
        schedule = RecordingSchedule(loss_signal=signal)
        observe_training_loss(
            schedule, time, prediction, tokens, objective, mask, step=12, training=True
        )
        observed_time, loss, step = schedule.observations[0]
        assert observed_time.tolist() == [0.25] and step == 12
        assert loss.item() == pytest.approx(expected)
        observe_training_loss(
            schedule, time, prediction, tokens, objective, mask, step=13, training=False
        )
        assert len(schedule.observations) == 1
        # An empty local batch must still participate in distributed gathering.
        observe_training_loss(
            schedule,
            time,
            prediction,
            tokens,
            objective,
            mask * 0,
            step=14,
            training=True,
        )
        assert schedule.observations[-1][0].numel() == 0


def test_score_parameterization_rejects_cross_entropy_feedback():
    predictor = AbsorbingScoreParameterization()
    validate_adaptive_feedback(RecordingSchedule(), predictor)
    with pytest.raises(TypeError, match="normalized token predictions"):
        validate_adaptive_feedback(
            RecordingSchedule(loss_signal="cross_entropy"), predictor
        )


@pytest.fixture(scope="module")
def calibration_cache(tmp_path_factory):
    path = tmp_path_factory.mktemp("adaptive_duo") / "v12.npz"
    np.savez(path, **build_integral_table(12, num_points=1001))
    return str(path)


@pytest.mark.parametrize(
    "method,step",
    [
        ("smoke", 7),
        ("uniform_diffusion", 7),
        ("sedd_absorb", 7),
        ("gaussian_diffusion_sudoku", 7),
        ("duo", 0),
        ("duo", 3),
    ],
)
def test_algorithm_feedback_preserves_loss_and_training_context(
    method, step, calibration_cache
):
    cfg = mdlm_config() if method == "smoke" else config(method, calibration_cache)
    algorithm = instantiate(cfg.algorithm)
    tokens = torch.randint(0, 9, (2, 10))
    mask = torch.ones_like(tokens)
    mask[:, :3] = 0
    torch.manual_seed(41)
    baseline = algorithm.loss(tokens, mask, step=step, training=True)
    schedule = RecordingSchedule(algorithm.schedule)
    algorithm.schedule = schedule
    torch.manual_seed(41)
    actual = algorithm.loss(tokens, mask, step=step, training=True)
    torch.testing.assert_close(actual["loss"], baseline["loss"])
    assert len(schedule.observations) == 1
    times, observed_loss, observed_step = schedule.observations[0]
    assert times.shape == observed_loss.shape == (2,)
    assert observed_step == step
    torch.testing.assert_close(observed_loss.mean(), actual["loss"].detach())
    actual["loss"].backward()
    assert all(
        torch.isfinite(p.grad).all()
        for p in algorithm.parameters()
        if p.grad is not None
    )
    algorithm.loss(tokens, mask, step=step, training=False)
    assert len(schedule.observations) == 1


@pytest.mark.parametrize("family", ["sflm", "cdcd"])
@pytest.mark.parametrize("continuous", [False, True])
def test_hydra_adaptive_training_sampling_and_resume(family, continuous):
    name = family + ("_gaussian_adaptive" if continuous else "_adaptive")
    method = "gaussian_diffusion_sudoku" if continuous else "uniform_diffusion"
    cfg = config(
        method,
        None,
        f"noise_schedule={name}",
        "noise_schedule.buffer_size=4",
        "noise_schedule.refit_every=1",
        "noise_schedule.warmup_steps=0",
        "noise_schedule.n_grid=101",
        "noise_schedule.n_knots=4",
    )
    algorithm = instantiate(cfg.algorithm)
    optimizer = torch.optim.Adam(algorithm.parameters(), lr=1e-4)
    tokens = torch.randint(0, 9, (2, 10))
    mask = torch.ones_like(tokens)
    mask[:, :3] = 0
    for step in range(3):
        optimizer.zero_grad()
        result = algorithm.loss(tokens, mask, step=step, training=True)
        assert torch.isfinite(result["loss"])
        result["loss"].backward()
        optimizer.step()
    clock = algorithm.schedule.clock if continuous else algorithm.schedule
    assert clock.refit_count.item() == 2
    before = {k: v.clone() for k, v in algorithm.schedule.state_dict().items()}
    algorithm.loss(tokens, mask, training=False)
    samples, stats = algorithm.generate(instantiate(cfg.sampler), tokens, ~mask.bool())
    assert samples.shape == tokens.shape and stats["nfe"] > 0
    assert torch.equal(samples[:, :3], tokens[:, :3])
    for key, value in algorithm.schedule.state_dict().items():
        torch.testing.assert_close(value, before[key])
    restored = instantiate(cfg.algorithm)
    restored.load_state_dict(algorithm.state_dict())
    algorithm.eval()
    restored.eval()
    torch.manual_seed(51)
    expected = algorithm.loss(tokens, mask)["loss"]
    torch.manual_seed(51)
    actual = restored.loss(tokens, mask)["loss"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    restored.train()
    restored.loss(tokens, mask, step=3, training=True)
    restored_clock = restored.schedule.clock if continuous else restored.schedule
    assert restored_clock.refit_count.item() == 3


def test_hydra_score_adaptation_requires_objective_signal():
    from hydra.errors import InstantiationException

    cfg = config("sedd_absorb", None, "noise_schedule=cdcd_adaptive")
    with pytest.raises(InstantiationException, match="normalized token predictions"):
        instantiate(cfg.algorithm)
    cfg.noise_schedule.loss_signal = "objective"
    instantiate(cfg.algorithm)

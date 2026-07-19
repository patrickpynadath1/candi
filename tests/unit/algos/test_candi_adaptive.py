import numpy as np
import pytest
import torch

from algos.candi_adaptive import (
    BezierCurve,
    JointLossSurface,
    PowerCurve,
    SequentialCurve,
    joint_surface_svg,
)


@pytest.mark.parametrize(
    "curve",
    [
        PowerCurve("linear", 1),
        PowerCurve("continuous_first", 2),
        PowerCurve("continuous_first", 8),
        PowerCurve("discrete_first", 4),
        BezierCurve((0.1, 0.7), (0.5, 0.95)),
        SequentialCurve(0.7, 0.3, 10),
    ],
)
def test_corruption_curves_are_monotone_with_exact_endpoints(curve):
    s = torch.linspace(0, 1, 101)
    discrete, continuous, d_discrete, d_continuous = curve(s)
    assert discrete[0] == pytest.approx(0)
    assert continuous[0] == pytest.approx(0)
    assert discrete[-1] == pytest.approx(1)
    assert continuous[-1] == pytest.approx(1)
    assert torch.all(discrete[1:] >= discrete[:-1])
    assert torch.all(continuous[1:] >= continuous[:-1])
    assert torch.all(d_discrete >= 0)
    assert torch.all(d_continuous >= 0)


def test_joint_loss_surface_tracks_2d_statistics_and_gradients():
    surface = JointLossSurface(bins=4, smoothing_sigma=0.5, min_effective_count=0.01)
    discrete = torch.tensor([0.1, 0.1, 0.6, 0.9])
    continuous = torch.tensor([0.1, 0.1, 0.6, 0.9])
    loss = discrete + 2 * continuous
    surface.record(discrete, continuous, loss)
    snapshot = surface.snapshot()
    assert snapshot["updates"] == 1
    assert np.asarray(snapshot["count"]).sum() == 4
    assert np.asarray(snapshot["count"])[0, 0] == 2
    assert np.asarray(snapshot["smoothed_loss"]).shape == (4, 4)
    assert np.asarray(snapshot["grad_discrete"]).shape == (4, 4)
    assert np.asarray(snapshot["grad_continuous"]).shape == (4, 4)
    assert np.asarray(snapshot["reliable"]).any()
    rendered = joint_surface_svg(snapshot, size=300, arrow_stride=2)
    assert rendered.startswith("<svg")
    assert "discrete corruption d" in rendered
    assert "continuous degradation c" in rendered
    labeled = joint_surface_svg(
        snapshot, size=300, arrow_stride=2, metric_name="raw ce"
    )
    assert "Smoothed raw ce and gradient field" in labeled


def test_bezier_rejects_nonmonotone_controls():
    with pytest.raises(ValueError, match="monotone"):
        BezierCurve((0.8, 0.2), (0.3, 0.9))


def test_extreme_power_curve_keeps_both_axes_numerically_active():
    curve = PowerCurve("continuous_first", 8, linear_mix=0.01)
    discrete, continuous, d_discrete, d_continuous = curve(torch.tensor([1e-3]))
    assert discrete.item() >= 1e-5
    assert continuous.item() > discrete.item()
    assert d_discrete.item() >= 0.0099
    assert d_continuous.item() >= 0.0099


@pytest.mark.parametrize(
    "curve",
    [
        PowerCurve("continuous_first", 4),
        PowerCurve("discrete_first", 2),
        BezierCurve((0.05, 0.55), (0.45, 0.95)),
        SequentialCurve(0.7, 0.3, 10),
    ],
)
def test_curve_derivatives_match_finite_differences(curve):
    s = torch.linspace(0.1, 0.9, 9, dtype=torch.float64)
    eps = 1e-5
    discrete, continuous, d_discrete, d_continuous = curve(s)
    discrete_plus, continuous_plus, _, _ = curve(s + eps)
    assert torch.allclose((discrete_plus - discrete) / eps, d_discrete, atol=2e-4)
    assert torch.allclose(
        (continuous_plus - continuous) / eps, d_continuous, atol=2e-4
    )

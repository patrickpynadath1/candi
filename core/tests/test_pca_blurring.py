from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from core.noise_schedules.linear_interpolation import LinearInterpolationSchedule
from core.processes.pca_blurring import PCABlurringProcess


def test_endpoints_mean_mask_and_gradients():
    torch.manual_seed(7)
    x = (torch.randn(3, 16, 8) + 7).requires_grad_()
    mask = torch.ones(3, 16, dtype=torch.bool)
    mask[:, :3] = False
    noise = torch.randn_like(x)
    p, schedule = PCABlurringProcess(), LinearInterpolationSchedule()
    for time in [0.0, 1.0]:
        y = p.corrupt(x, schedule(torch.full((3,), time)), mask, noise=noise)
        torch.testing.assert_close(
            y, torch.where(mask[..., None], noise, x) if time else x, rtol=0, atol=0
        )
    y = p.corrupt(x, schedule(torch.full((3,), 0.4)), mask, noise=noise)
    y.square().mean().backward()
    assert x.grad.isfinite().all() and x.grad.abs().sum() > 0
    assert not p.basis(x, mask).vectors.requires_grad
    # Constant sequences have zero centered covariance but their mean is attenuated.
    const = torch.full_like(x, 5)
    torch.testing.assert_close(
        p.corrupt(
            const, schedule(torch.full((3,), 0.4)), mask, noise=torch.zeros_like(x)
        ),
        torch.where(mask[..., None], 0.6 * const, const),
    )


def test_rotation_equivariance_and_isotropic_limit():
    torch.manual_seed(2)
    x = torch.randn(2, 32, 6)
    mask = torch.ones(2, 32, dtype=torch.bool)
    q = torch.linalg.qr(torch.randn(6, 6)).Q
    schedule = LinearInterpolationSchedule()(torch.full((2,), 0.4))
    p = PCABlurringProcess()
    y = p.corrupt(x, schedule, mask, noise=torch.zeros_like(x))
    rotated = p.corrupt(x @ q, schedule, mask, noise=torch.zeros_like(x))
    torch.testing.assert_close(rotated, y @ q, atol=2e-5, rtol=2e-5)
    p = PCABlurringProcess(strength=0)
    torch.testing.assert_close(
        p.corrupt(x, schedule, mask, noise=torch.zeros_like(x)),
        0.6 * x,
        atol=2e-6,
        rtol=2e-6,
    )


def test_dominant_modes_degenerate_spaces_and_mask_exclusion():
    p = PCABlurringProcess()
    x = torch.cat([torch.eye(4), -torch.eye(4)])[None]
    mask = torch.ones(1, 8, dtype=torch.bool)
    assert torch.allclose(p.basis(x, mask).rates, p.basis(x, mask).rates[:, :1])
    x = x * torch.tensor([10.0, 2.0, 1.0, 0.0])
    basis = p.basis(x, mask)
    assert (basis.rates.diff(dim=-1) >= 0).all()
    assert basis.rates[0, -1] > basis.rates[0, 0]
    mask[:, :2] = False
    modified = x.clone()
    modified[:, :2] = 1e5
    torch.testing.assert_close(
        p.basis(x, mask).eigenvalues, p.basis(modified, mask).eigenvalues
    )
    for active in [
        torch.zeros_like(mask),
        torch.nn.functional.one_hot(torch.tensor([0]), 8).bool(),
    ]:
        assert p.basis(x, active).rates.isfinite().all()


def test_conditional_forward_moments_and_composition():
    torch.manual_seed(12)
    p, schedule = PCABlurringProcess(), LinearInterpolationSchedule()
    x = torch.randn(1, 9, 3).expand(6000, -1, -1).clone()
    mask = torch.ones(x.shape[:2], dtype=torch.bool)
    s, mid, t = [schedule(torch.full((len(x),), z)) for z in [0.2, 0.5, 0.8]]
    initial = p.corrupt(x, s, mask)
    direct = p.forward_transition(initial, x, s, t, mask)
    composed = p.forward_transition(
        p.forward_transition(initial, x, s, mid, mask), x, mid, t, mask
    )
    mean = p.corrupt(
        x[:1], schedule(torch.tensor([0.8])), mask[:1], noise=torch.zeros_like(x[:1])
    )
    for sample in [direct, composed]:
        torch.testing.assert_close(sample.mean(0), mean[0], atol=0.04, rtol=0)
        torch.testing.assert_close(
            sample.var(0), torch.full_like(mean[0], 0.64), atol=0.045, rtol=0
        )
    with pytest.raises(ValueError, match="negative variance"):
        p.forward_transition(initial, x, t, s, mask)


def test_sudoku_recipe_trains_and_renoises():
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=sudoku_pca_blurring",
                "model.backbone.hidden_size=32",
                "model.backbone.n_heads=2",
                "model.backbone.n_blocks=1",
                "model.backbone.cond_dim=16",
                "model.backbone.dropout=0",
                "model.views.content.representation.state_dim=8",
                "model.views.content.representation.state_representation.dimension=8",
                "sampler.steps=3",
                "time_sampler.warmup=0",
            ],
        )
    algo = instantiate(cfg.algorithm)
    tokens = torch.randint(0, 12, (2, 180))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, :90] = False
    result = algo.loss(tokens, mask, training=True, step=0)
    result["loss"].backward()
    assert result["loss"].isfinite()
    assert all(p.grad.isfinite().all() for p in algo.parameters() if p.grad is not None)
    algo.time_sampler.post_optimizer_step()
    assert algo.time_sampler.ready
    algo.eval()
    with torch.no_grad():
        generated, info = algo.generate(instantiate(cfg.sampler), tokens, ~mask)
    assert torch.equal(generated[~mask], tokens[~mask])
    assert (
        generated.shape == tokens.shape
        and generated.min() >= 0
        and generated.max() < 12
    )
    assert info["nfe"] == 4


def test_near_terminal_continuity_and_near_degenerate_rotation():
    p = PCABlurringProcess()
    x = torch.cat([torch.eye(4), -torch.eye(4)])[None]
    x[..., 0] *= 1.00001
    mask = torch.ones(1, 8, dtype=torch.bool)
    q = torch.linalg.qr(torch.randn(4, 4)).Q
    schedule = LinearInterpolationSchedule()
    path = schedule(torch.tensor([0.4]))
    a = p.corrupt(x, path, mask, noise=torch.zeros_like(x))
    b = p.corrupt(x @ q, path, mask, noise=torch.zeros_like(x))
    torch.testing.assert_close(b, a @ q, atol=2e-6, rtol=2e-5)
    noise = torch.randn_like(x)
    near = p.corrupt(x, schedule(torch.tensor([1 - 1e-5])), mask, noise=noise)
    torch.testing.assert_close(near, noise, atol=5e-5, rtol=0)

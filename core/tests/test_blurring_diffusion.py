import pytest
import torch

from core.algorithms.blurring_diffusion import BlurringDiffusion
from core.models.continuous_dit import ContinuousDiT
from core.modifiers.block_diffusion import BlockDiffusion
from core.modifiers.self_conditioning.prediction import PredictionCoordinator
from core.noise_schedules.blurring import BlurringSchedule
from core.objectives.cross_entropy import CrossEntropy
from core.parameterizations.posterior import PosteriorParameterization
from core.processes.blurring import BlurringProcess, dct_matrix
from core.representations.centered_embedding import CenteredEmbeddingRepresentation
from core.samplers.blur_ancestral import BlurAncestralSampler
from core.sequences import SequenceLayout, resolve_sequence_layout
from core.time_sampling.uniform import UniformTimeSampler


def make_algorithm(*, length=7, self_conditioning=False, source="posterior_mean"):
    representation = CenteredEmbeddingRepresentation(11, 4)
    model = ContinuousDiT(
        11,
        hidden_size=16,
        cond_dim=8,
        n_blocks=1,
        n_heads=2,
        dropout=0,
        state_dim=4,
        input_architecture="linear",
        input_projection_bias=True,
        input_preconditioning="none",
        time_conditioning="time",
        output_dim=12,
        self_conditioning=self_conditioning,
        self_conditioning_source=source,
    )
    return BlurringDiffusion(
        model,
        PosteriorParameterization(),
        BlurringProcess(),
        BlurringSchedule(),
        UniformTimeSampler(),
        CrossEntropy(),
        representation,
        length=length,
        prediction=PredictionCoordinator(
            enabled=self_conditioning,
            source=source,
            probability=1,
            evaluation_probability=1,
        ),
    )


def test_dct_and_schedule_endpoints_and_finite_derivatives():
    basis = dct_matrix(17, dtype=torch.float64)
    assert torch.allclose(
        basis.T @ basis, torch.eye(17, dtype=torch.float64), atol=1e-12
    )
    schedule = BlurringSchedule()
    coefficients = schedule.coefficients(
        torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64), 89, 16
    )
    assert torch.equal(
        coefficients.attenuation[0], torch.ones_like(coefficients.attenuation[0])
    )
    assert coefficients.sigma[0] == 0
    assert coefficients.sigma[-1] == 1
    assert torch.isfinite(coefficients.dattenuation_dt).all()
    assert torch.isfinite(coefficients.dsigma_dt).all()


def test_feature_order_rho_endpoints_and_singleton_are_finite():
    time = torch.tensor([0.6], dtype=torch.float64)
    joint = BlurringSchedule().coefficients(time, 6, 4).attenuation
    assert torch.all(joint[0, :, 0] > joint[0, :, -1])
    positional = BlurringSchedule(feature_rho=1).coefficients(time, 6, 4).attenuation
    modes = BlurringSchedule(feature_rho=1).mode_coefficients(time, 6)[0]
    assert torch.allclose(positional, modes[:, :, None].expand_as(positional))
    feature = BlurringSchedule(feature_rho=0).coefficients(time, 6, 4).attenuation
    assert torch.allclose(feature[:, :1], feature)
    singleton = BlurringSchedule().coefficients(torch.tensor([0.0, 1.0]), 1, 4)
    assert torch.isfinite(singleton.attenuation).all()
    assert torch.isfinite(singleton.dattenuation_dt).all()
    for rho in (0, 0.5, 1):
        schedule = BlurringSchedule(feature_rho=rho)
        schedule.validate_shape(32, 4)
        schedule.validate_shape(89, 4)


def test_layout_intersects_blocks_masks_and_corruption_is_standalone_equivalent():
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 1]], dtype=torch.bool)
    layout = SequenceLayout.fixed_blocks(1, 7, 3, mask.device)
    assert list(layout.spans(mask)) == [(0, 0, 3), (0, 3, 5), (0, 6, 7)]
    process, schedule = BlurringProcess(), BlurringSchedule()
    clean, noise = torch.randn(1, 7, 4), torch.randn(1, 7, 4)
    time = torch.tensor([0.55])
    path = schedule.for_layout(time, mask, 4, layout)
    joint = process.corrupt(clean, path, noise=noise)
    for start, end in [(0, 3), (3, 5), (6, 7)]:
        local_mask = torch.ones(1, end - start, dtype=torch.bool)
        local_layout = resolve_sequence_layout(local_mask)
        local_path = schedule.for_layout(time, local_mask, 4, local_layout)
        local = process.corrupt(
            clean[:, start:end], local_path, noise=noise[:, start:end]
        )
        assert torch.allclose(joint[:, start:end], local)
    changed = clean.clone()
    changed[:, :3] += 100
    changed_joint = process.corrupt(changed, path, noise=noise)
    assert torch.equal(joint[:, 3:], changed_joint[:, 3:])


def test_reverse_bridge_reblurs_oracle_and_matches_marginal_variance():
    process, schedule = BlurringProcess(), BlurringSchedule()
    length, width = 6, 3
    clean = torch.randn(2, length, width)
    time, target_time = torch.tensor([0.8, 0.6]), torch.tensor([0.35, 0.2])
    current = schedule.coefficients(time, length, width)
    target = schedule.coefficients(target_time, length, width)
    state = current.attenuation * clean
    mean, variance = process.reverse_mode_moments(state, clean, current, target)
    assert torch.allclose(mean, target.attenuation * clean, atol=2e-5)
    mean_shifted, _ = process.reverse_mode_moments(state + 1, clean, current, target)
    coefficient = mean_shifted - mean
    marginal = coefficient.square() * current.sigma[:, None, None].square() + variance
    assert torch.allclose(
        marginal, target.sigma[:, None, None].square().expand_as(marginal), atol=2e-6
    )


def test_reverse_spans_are_independent_with_controlled_noise():
    process, schedule = BlurringProcess(), BlurringSchedule()
    mask = torch.ones(1, 6, dtype=torch.bool)
    layout = SequenceLayout.fixed_blocks(1, 6, 3, mask.device)
    current = schedule.for_layout(torch.tensor([0.7]), mask, 4, layout)
    target = schedule.for_layout(torch.tensor([0.3]), mask, 4, layout)
    state, clean, noise = (
        torch.randn(1, 6, 4),
        torch.randn(1, 6, 4),
        torch.randn(1, 6, 4),
    )
    joint = process.reverse(state, clean, current, target, noise=noise)
    changed_state, changed_clean = state.clone(), clean.clone()
    changed_state[:, :3] += 100
    changed_clean[:, :3] -= 100
    changed = process.reverse(
        changed_state, changed_clean, current, target, noise=noise
    )
    assert torch.equal(joint[:, 3:], changed[:, 3:])
    for start, end in ((0, 3), (3, 6)):
        local_mask = torch.ones(1, 3, dtype=torch.bool)
        local_layout = resolve_sequence_layout(local_mask)
        local_current = schedule.for_layout(
            torch.tensor([0.7]), local_mask, 4, local_layout
        )
        local_target = schedule.for_layout(
            torch.tensor([0.3]), local_mask, 4, local_layout
        )
        local = process.reverse(
            state[:, start:end],
            clean[:, start:end],
            local_current,
            local_target,
            noise=noise[:, start:end],
        )
        assert torch.allclose(joint[:, start:end], local)


def test_representation_projection_gradient_and_detach():
    representation = CenteredEmbeddingRepresentation(13, 4)
    tokens = torch.arange(7)[None]
    representation.encode(tokens).sum().backward()
    assert representation.table.grad is not None
    representation.table.grad = None
    assert representation.encode(tokens).requires_grad
    with torch.no_grad():
        representation.table.add_(0.2)
    representation.post_optimizer_step()
    table = representation.codebook()
    assert torch.allclose(table.norm(dim=-1), torch.ones(13), atol=1e-6)
    assert torch.allclose(table.mean(0), torch.zeros(4), atol=3e-3)


@pytest.mark.parametrize("source", ["posterior_mean", "hidden"])
def test_algorithm_backward_block_training_and_sampler_nfe(source):
    algorithm = make_algorithm(length=7, self_conditioning=True, source=source)
    tokens = torch.randint(0, 11, (2, 7))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    wrapped = BlockDiffusion(
        algorithm, block_size=3, clean_loss="autoregressive", cache_clean_prefix=False
    )
    result = wrapped.loss(tokens, mask, training=True)
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert algorithm.representation.table.grad is not None
    samples, metadata = wrapped.generate(
        BlurAncestralSampler(steps=2),
        tokens,
        torch.tensor([[1, 1, 0, 0, 0, 0, 0]] * 2, dtype=torch.bool),
    )
    assert samples.shape == tokens.shape
    assert torch.equal(samples[:, :2], tokens[:, :2])
    assert metadata["nfe"] == 9


def test_self_conditioning_selects_spans_not_batch_dimension():
    from core.modifiers.self_conditioning.prediction import _select_batch
    from core.modifiers.hybrid.views import ViewPoint
    from core.noise_schedules.blurring import BlurringSchedule
    from core.sequences import SequenceLayout

    # Unequal span counts and lengths: neither group dimension equals batch size.
    mask = torch.tensor(
        [[1, 1, 0, 1, 1, 1], [0, 1, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0]], dtype=torch.bool
    )
    time = torch.tensor([0.2, 0.4, 0.6])
    schedule = BlurringSchedule(terminal_mean=0.01)
    path = schedule.for_layout(time, mask, 8, SequenceLayout.contiguous(3, 6, "cpu"))
    for selected in [
        torch.tensor([True, False, True]),
        torch.tensor([False, True, False]),
        torch.zeros(3, dtype=torch.bool),
    ]:
        result = _select_batch(ViewPoint(time, path), selected)
        expected = schedule.for_layout(
            time[selected],
            mask[selected],
            8,
            SequenceLayout.contiguous(int(selected.sum()), 6, "cpu"),
        )
        assert [(g.length, g.spans) for g in result.coefficients.groups] == [
            (g.length, g.spans) for g in expected.groups
        ]
        for actual, target in zip(result.coefficients.groups, expected.groups):
            torch.testing.assert_close(
                actual.coefficients.attenuation, target.coefficients.attenuation
            )
            torch.testing.assert_close(
                actual.coefficients.sigma, target.coefficients.sigma
            )

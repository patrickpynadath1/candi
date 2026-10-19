"""Audit the mathematical consequences of changing the TinyGSM DDP partition."""

import torch
import pytest
from core.time_sampling.uniform import UniformTimeSampler
from core.time_sampling.loss_derivative import LossDerivativeTimeSampler
from core.algorithms.discrete import reduce_token_loss

LAYOUTS = [(8, 8, 8), (4, 8, 16), (4, 16, 8)]  # ranks, accumulation, microbatch


@pytest.mark.parametrize("world,accum,micro", LAYOUTS)
def test_all_layouts_cover_every_antithetic_stratum(world, accum, micro):
    sampler = UniformTimeSampler()
    samples = []
    for rank in range(world):
        for index in range(accum):
            samples.append(
                sampler.sample(
                    micro,
                    "cpu",
                    global_batch_size=512,
                    world_size=world,
                    rank=rank,
                    accumulation_steps=accum,
                    accumulation_index=index,
                )
            )
    samples = torch.cat(samples)
    strata = torch.floor(
        (samples - sampler.epsilon) / (1 - sampler.epsilon) * 512
    ).long()
    assert torch.equal(strata.sort().values, torch.arange(512))


def accumulated_gradient(layout, variable_counts):
    world, accum, micro = layout
    parameter = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    # DistributedSampler assigns rank-strided samples from the global permutation.
    ids = torch.randperm(512, generator=torch.Generator().manual_seed(22))
    counts = (torch.arange(512) % 7 + 1) if variable_counts else torch.full((512,), 7)
    signal = counts.double().square() + torch.arange(512).double() / 100
    for rank in range(world):
        for part in ids[rank::world].reshape(accum, micro):
            mask = torch.arange(7)[None] < counts[part, None]
            loss = reduce_token_loss(
                parameter * signal[part, None].expand(-1, 7), mask
            )["loss"]
            (loss / (world * accum)).backward()
    return parameter.grad.item()


def test_equal_token_counts_have_equivalent_accumulated_gradients():
    gradients = [accumulated_gradient(layout, False) for layout in LAYOUTS]
    torch.testing.assert_close(torch.tensor(gradients), torch.full((3,), gradients[0]))


def test_variable_token_counts_change_weights_even_at_same_microbatch_size():
    gradients = [accumulated_gradient(layout, True) for layout in LAYOUTS]
    assert abs(gradients[0] - gradients[1]) > 1e-3
    assert abs(gradients[0] - gradients[2]) > 1e-3


def refit_audit(layout):
    world, accum, micro = layout
    sampler = LossDerivativeTimeSampler(adaptive=True, warmup=0, refit_every=50)
    seen = []
    sampler.refit = lambda: seen.append(int(sampler.count))
    # Model the concatenated all-gather result from each microbatch round.
    # observe sees the same pairs/count as one call containing all rank inputs.
    for _ in range(accum):
        size = world * micro
        sampler.observe(
            torch.linspace(0.01, 0.99, size), torch.ones(size, 2), None, step=1000
        )
    return seen, int(sampler.count)


def test_refit_occurs_once_but_before_full_optimizer_batch():
    results = [refit_audit(layout) for layout in LAYOUTS]
    assert results == [([64], 512), ([64], 512), ([32], 512)]


def test_validation_interval_is_in_microbatches():
    # Lightning val_check_interval counts training batches, unlike max_steps,
    # scheduler/EMA, checkpoint.every_n_steps and TaskEvaluation cadence.
    assert 10000 // 8 == 1250
    assert 10000 // 16 == 625
    assert 20000 // 16 == 1250


def test_refitted_distribution_depends_on_layout_even_with_same_observation_count():
    cdfs = []
    for world, accum, micro in LAYOUTS:
        times = (torch.arange(512).float() + 0.5) / 512
        first_round = times.reshape(world, accum, micro)[:, 0].reshape(-1)
        sampler = LossDerivativeTimeSampler(adaptive=True, warmup=0, refit_every=50)
        losses = first_round.square() + first_round.pow(3)
        sampler.observe(first_round, losses[:, None], None, step=1000)
        assert sampler.ready
        cdfs.append(sampler.cdf)
    assert not torch.allclose(cdfs[0], cdfs[1])
    assert not torch.allclose(cdfs[0], cdfs[2])

"""Every optimizer step must see a full set of stratified time partitions."""

from core.data.module import WholeStepSampler


def sampler(rank, epoch=0):
    result = WholeStepSampler(
        range(29), num_replicas=2, rank=rank, step_examples=4, seed=7
    )
    result.set_epoch(epoch)
    return result


def test_each_rank_gets_whole_disjoint_steps():
    first, second = list(sampler(0)), list(sampler(1))
    # 29 // 2 = 14 per rank, trimmed to 12: three whole steps of four examples.
    assert len(first) == len(second) == len(sampler(0)) == 12
    assert not set(first) & set(second)


def test_epochs_reshuffle_reproducibly():
    assert list(sampler(0, 0)) == list(sampler(0, 0))
    assert list(sampler(0, 0)) != list(sampler(0, 1))
    # The trimmed tail differs per epoch, so no example is permanently dropped.
    covered = set()
    for epoch in range(20):
        covered |= set(sampler(0, epoch)) | set(sampler(1, epoch))
    assert covered == set(range(29))

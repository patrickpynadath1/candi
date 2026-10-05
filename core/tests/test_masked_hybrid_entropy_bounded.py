import pytest
import torch
from core.tests.test_hybrid_composition import compose_hybrid
from core.samplers.hybrid_entropy_bounded import MaskedHybridEntropyBoundedSampler


@pytest.mark.parametrize('first,second', [('gaussian', 'masked'), ('masked', 'gaussian')])
@pytest.mark.parametrize('feedback', ['none', 'hidden'])
def test_eb_hybrid_order_and_prompt(first, second, feedback):
    algo, _ = compose_hybrid(first, second, feedback)
    sampler = MaskedHybridEntropyBoundedSampler(entropy_bound=0, steps=6)
    tokens = torch.randint(0, 12, (2, 6))
    prompt = torch.zeros_like(tokens, dtype=torch.bool)
    prompt[0] = True
    prompt[1, 0] = True
    generated, info = algo.generate(sampler, tokens, prompt)
    assert torch.equal(generated[prompt], tokens[prompt])
    assert generated.min() >= 0 and generated.max() < 12
    assert info['nfe_per_sample'][0] == 0
    assert 1 <= info['nfe_per_sample'][1] <= 5
    assert sum(row[1] for row in info['accepted_per_step']) == 5


def test_requires_masked_view():
    algo, _ = compose_hybrid('gaussian', 'uniform')
    tokens = torch.zeros(1, 3, dtype=torch.long)
    with pytest.raises(ValueError, match='absorbing'):
        algo.generate(MaskedHybridEntropyBoundedSampler(), tokens, tokens.bool())


def test_eb_uses_source_global_clock_and_configured_continuous_update():
    from unittest.mock import patch
    algo, _ = compose_hybrid('gaussian', 'masked')
    tokens = torch.randint(0, 12, (1, 4))
    prompt = torch.zeros_like(tokens, dtype=torch.bool)
    update = algo.views[0].update
    with patch.object(algo, 'sampling_prediction', wraps=algo.sampling_prediction) as predict:
        with patch.object(algo.views[0], 'update', wraps=update) as continuous:
            with patch.object(algo.views[1], 'update', side_effect=AssertionError('EB owns mask reveals')):
                _, info = algo.generate(MaskedHybridEntropyBoundedSampler(entropy_bound=0, steps=4), tokens, prompt)
    assert continuous.call_count == predict.call_count == info['nfe']
    grid = torch.linspace(1, 1e-5, 5)
    for index, (prediction, transition) in enumerate(zip(predict.call_args_list, continuous.call_args_list)):
        states, points, time = prediction.args
        assert torch.allclose(time, grid[index].expand(1))
        assert torch.allclose(points[0].time, time)
        assert torch.allclose(transition.args[4].time, time)
        assert torch.allclose(transition.args[5].time, grid[index + 1].expand(1))

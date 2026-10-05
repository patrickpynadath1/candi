import pytest
import torch
from core.tests.test_hybrid_composition import compose_hybrid
from core.samplers.parallel_entropy_bounded import ParallelEntropyBoundedSampler
from core.samplers.hybrid_entropy_bounded import MaskedHybridEntropyBoundedSampler

@pytest.mark.parametrize('family', ['gaussian', 'spherical', 'simplex'])
@pytest.mark.parametrize('freeze', [False, True])
@pytest.mark.parametrize('feedback', ['none', 'hidden', 'token'])
def test_uniform_parallel_eb(family, freeze, feedback):
    algo, _ = compose_hybrid(family, 'uniform', 'hidden' if feedback == 'hidden' else 'none')
    if feedback == 'token':
        from core.modifiers.token_feedback.training import TokenFeedback
        algo = TokenFeedback(algo, depth=0)
    tokens=torch.randint(0,12,(2,5)); mask=torch.zeros_like(tokens,dtype=torch.bool)
    mask[0]=True;mask[1,0]=True
    samples,info=algo.generate(ParallelEntropyBoundedSampler(freeze=freeze,steps=5),tokens,mask)
    assert torch.equal(samples[mask],tokens[mask])
    assert samples.min()>=0 and samples.max()<12
    assert info['nfe_per_sample'][0]==0
    assert 1<=info['nfe_per_sample'][1]<=6

@pytest.mark.parametrize('family', ['gaussian','spherical','simplex'])
def test_frozen_matches_existing_masked_sampler(family):
    algo,_=compose_hybrid(family,'masked')
    tokens=torch.randint(0,12,(2,5));mask=torch.zeros_like(tokens,dtype=torch.bool)
    torch.manual_seed(9)
    a,ma=algo.generate(MaskedHybridEntropyBoundedSampler(steps=5),tokens,mask)
    torch.manual_seed(9)
    b,mb=algo.generate(ParallelEntropyBoundedSampler(steps=5),tokens,mask)
    assert torch.equal(a,b)
    assert ma['nfe_per_sample']==mb['nfe_per_sample']


def test_released_gaussian_commitment_is_forward_corrupted(monkeypatch):
    from unittest.mock import Mock
    import core.samplers.parallel_entropy_bounded as eb
    algo,_=compose_hybrid('gaussian','uniform')
    view=algo.views[0]
    original=view.process.corrupt_state
    calls=[]
    def corrupt(clean,coefficients,mask):
        calls.append((clean.clone(),coefficients,mask.clone()))
        return original(clean,coefficients,mask)
    monkeypatch.setattr(view.process,'corrupt_state',corrupt)
    selections=[torch.tensor([[False,True,False]]),
                torch.tensor([[False,False,True]])]
    monkeypatch.setattr(eb,'select_prefix',Mock(side_effect=selections))
    tokens=torch.tensor([[1,2,3]])
    sample,info=algo.generate(eb.ParallelEntropyBoundedSampler(freeze=False,steps=2),
                              tokens,torch.tensor([[True,False,False]]))
    assert len(calls)==1
    assert calls[0][2].tolist()==[[False,True,False]]
    assert sample[0,0]==1
    assert info['completed_per_sample']==[False]
    assert info['nfe_per_sample']==[3]

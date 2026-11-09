from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from hydra.utils import instantiate
from core.tests.test_replaid_hybrid import config
from core.samplers.churn import RandomUncommitment
from core.samplers.parallel_entropy_bounded import ParallelEntropyBoundedSampler


def test_reopening_resets_both_views_and_preserves_other_positions():
    algo = instantiate(config().algorithm).eval()
    tokens = torch.tensor([[1,2,3,4]])
    prompt = torch.tensor([[True,False,False,False]])
    committed = torch.tensor([[True,True,False,True]])
    states = tuple(v.representation.encode(tokens) for v in algo.views)
    points = algo.points(torch.tensor([.5]), corruption_mask=~prompt)
    churn = RandomUncommitment(rate=1)
    churn.validate(algo.views, 'freeze')
    new, kept, released = churn(algo.views,states,points,tokens,committed,prompt,0,5)
    assert released.tolist() == [[False,True,False,True]]
    assert kept.tolist() == [[True,False,False,False]]
    assert (new[1][released] == algo.views[1].process.mask_index).all()
    for before, after in zip(states,new):
        assert torch.equal(before[~released],after[~released])
    assert not torch.equal(new[0][released],states[0][released])


def test_zero_churn_preserves_rng_trajectory_and_early_stop():
    algo = instantiate(config().algorithm).eval()
    tokens = torch.tensor([[1,2,3,4]])
    prompt = torch.tensor([[True,False,False,False]])
    outputs=[]
    for churn in [None,RandomUncommitment(0)]:
        sampler = ParallelEntropyBoundedSampler(steps=5,entropy_bound=100,churn=churn)
        torch.manual_seed(17)
        out,info=algo.generate(sampler,tokens,prompt)
        outputs.append((out,info,torch.get_rng_state()))
    assert torch.equal(outputs[0][0],outputs[1][0])
    assert torch.equal(outputs[0][2],outputs[1][2])
    assert outputs[0][1]['nfe']==outputs[1][1]['nfe']==1


def test_refinement_defers_completion_and_counts_real_forwards():
    algo=instantiate(config().algorithm).eval()
    tokens=torch.tensor([[1,2,3,4],[1,2,3,4]])
    prompt=torch.tensor([[True,False,False,False],[True,True,True,True]])
    sampler=ParallelEntropyBoundedSampler(steps=5,entropy_bound=100,
        churn=RandomUncommitment(rate=1,stop_fraction=.8))
    with patch.object(algo.model,'forward',wraps=algo.model.forward) as calls:
        out,info=algo.generate(sampler,tokens,prompt)
    assert info['nfe']==calls.call_count==5
    assert torch.equal(out[prompt],tokens[prompt])
    assert info['nfe_per_sample']==[5,0]
    assert sum(row[0] for row in info['uncommitted_per_step'])>0
    assert all(row[1]==0 for row in info['uncommitted_per_step'])
    assert info['uncommitted_per_step'][-1]==[0,0]


@pytest.mark.parametrize('rate',[-.1,1.1,float('nan')])
def test_invalid_rates(rate):
    with pytest.raises(ValueError):
        RandomUncommitment(rate)


@pytest.mark.parametrize('family',['simplex','spherical','gaussian','blurring','uniform'])
@pytest.mark.parametrize('hybrid',[False,True])
def test_process_geometry_and_churn(family, hybrid):
    from core.sweeps.configuration import compose
    from core.tests.networks import TINY16
    from omegaconf import OmegaConf
    cfg=compose('corruption_factorial',[
        f'information_path/hybrid_view@model.views.content={family}',
        *TINY16,'task.vocab_size=12','task.length=8','training.ema=0'])
    if hybrid:
        OmegaConf.update(cfg,'model.views.discrete',OmegaConf.load(
            'configs/information_path/hybrid_view/masked.yaml'),force_add=True)
    algo=instantiate(cfg.algorithm).eval()
    assert len(algo.views) == (2 if hybrid else 1)
    tokens=torch.randint(0,12,(2,8))
    prompt=torch.zeros_like(tokens,dtype=torch.bool)
    prompt[:,0]=True
    points=algo.points(torch.full((2,),.5),corruption_mask=~prompt)
    states=tuple(v.representation.clean_state(tokens) for v in algo.views)
    churn=RandomUncommitment(rate=1)
    churn.validate(algo.views,'freeze')
    new,committed,released=churn(algo.views,states,points,tokens,
        torch.ones_like(prompt),prompt,0,5)
    assert torch.equal(released,~prompt)
    for before,after in zip(states,new):
        assert torch.equal(before[prompt],after[prompt])
    content=new[0][released]
    if family=='simplex':
        assert torch.isfinite(content).all()
        torch.testing.assert_close(content.exp().sum(-1),torch.ones_like(content[:,0]))
    elif family=='spherical':
        torch.testing.assert_close(content.norm(dim=-1),torch.ones_like(content[:,0]))
    if hybrid:
        assert (new[1][released]==algo.views[1].process.mask_index).all()
    sampler=ParallelEntropyBoundedSampler(steps=5,entropy_bound=100,churn=churn)
    with patch.object(algo.model,'forward',wraps=algo.model.forward) as calls:
        out,info=algo.generate(sampler,tokens,prompt)
    assert torch.equal(out[prompt],tokens[prompt])
    assert info['nfe']==calls.call_count==5
    assert ((out>=0)&(out<12)).all()


def test_missing_forward_kernel_is_rejected_by_capability():
    churn=RandomUncommitment()
    view=SimpleNamespace(process=object(),representation=SimpleNamespace(encode=lambda x:x))
    with pytest.raises(TypeError,match='corrupt_state'):
        churn.validate([view],'freeze')

"""Score temperature is noise rescaling, independent of categorical temperature."""
import pytest
import torch
from hydra.utils import instantiate
from core.sweeps.configuration import compose
from core.noise_schedules.gaussian_path import path_from_gamma
from core.samplers.score_temperature import GaussianScoreTemperature
from core.samplers.policies import ReverseTransition
from core.training.module import TrainingModule


@pytest.mark.parametrize('temperature', [0.5, 0.9, 1.0, 1.5])
def test_matches_plaid_noise_rescaling(temperature):
    torch.manual_seed(4)
    state, clean = [torch.randn(2, 3, 4, dtype=torch.float64) for _ in range(2)]
    path = path_from_gamma(torch.tensor([-3., 5.], dtype=torch.float64), torch.ones(2))
    a, s = path.alpha[:, None, None], path.sigma[:, None, None]
    epsilon = (state-a*clean)/s
    expected = (state-s*epsilon/temperature)/a
    result = GaussianScoreTemperature(temperature)(state, clean, path)
    torch.testing.assert_close(result, expected)
    if temperature == 1:
        assert result is clean


@pytest.mark.parametrize('temperature', [0., -1., float('nan'), float('inf')])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError, match='finite and positive'):
        GaussianScoreTemperature(temperature)


def test_pure_noise_endpoint_rejected_only_when_tempered():
    from types import SimpleNamespace
    x = torch.ones(1, 2, 3)
    path = SimpleNamespace(alpha=torch.zeros(1))
    assert GaussianScoreTemperature()(x, x, path) is x
    with pytest.raises(ValueError, match='positive source alpha'):
        GaussianScoreTemperature(.9)(x, x, path)


def test_discrete_only_transition_rejected():
    a = instantiate(compose('mdlm', ['task.vocab_size=5', 'model.hidden_size=32',
                                     'model.cond_dim=16', 'model.n_blocks=1', 'model.n_heads=4']).algorithm)
    with pytest.raises(TypeError, match='Gaussian posterior'):
        ReverseTransition(score_temperature=.9).validate(a.views, 'none')


@pytest.mark.parametrize('recipe,key', [('replaid','sampler.score_temperature'),
                                       ('replaid_hybrid','sampler.transition.score_temperature')])
def test_native_sampler_prompt_nfe_and_update(recipe, key):
    cfg = compose(recipe, ['task.vocab_size=5', 'task.length=4', 'model.hidden_size=32',
                          'model.cond_dim=16', 'model.n_blocks=1', 'model.n_heads=4',
                          'model.dropout=0.0', 'sampler.steps=2', key+'=0.9'])
    a = TrainingModule(cfg).algorithm.eval()
    calls=[]
    hook=a.model.register_forward_hook(lambda *args: calls.append(1))
    sampler=instantiate(cfg.sampler)
    x=torch.tensor([[1,2,3,4]])
    fixed=torch.tensor([[True,False,False,False]])
    torch.manual_seed(123)
    result,info=a.generate(sampler,x,fixed)
    hook.remove()
    assert torch.equal(result[fixed],x[fixed])
    assert info['nfe']==len(calls)
    assert sampler.steps_for_nfe(info['nfe'])==2
    assert result.min()>=0 and result.max()<5


@pytest.mark.parametrize('recipe,policy,key', [
    ('replaid', 'gaussian_score_temperature', 'sampler.score_temperature'),
    ('replaid_hybrid', 'hybrid_score_temperature', 'sampler.transition.score_temperature'),
])
def test_tuning_policy_and_effect_on_first_transition(recipe, policy, key):
    from omegaconf import OmegaConf
    cfg = compose(recipe, ['+inference/inference_policies='+policy,
        'task.vocab_size=5', 'task.length=4', 'model.hidden_size=32',
        'model.cond_dim=16', 'model.n_blocks=1', 'model.n_heads=4',
        'model.dropout=0.0', 'sampler.steps=2'])
    policy_cfg = cfg.preparation.inference[0]
    assert key in policy_cfg.space
    a = TrainingModule(cfg).algorithm.eval()
    process = a.process if recipe == 'replaid' else a.views[0].process
    original = process.reverse_posterior
    captured = []
    def reverse(state, clean, source, destination, noise):
        captured.append((state.clone(), clean.clone(), source, noise.clone()))
        return original(state, clean, source, destination, noise)
    process.reverse_posterior = reverse
    x = torch.tensor([[1,2,3,4]])
    fixed = torch.tensor([[True,False,False,False]])
    outputs = []
    for tau in [1., .9]:
        OmegaConf.update(cfg, key, tau)
        sampler = instantiate(cfg.sampler)
        captured.clear()
        torch.manual_seed(41)
        a.generate(sampler, x, fixed)
        outputs.append(captured[0])
    state, clean, path, noise = outputs[0]
    tempered_state, tempered_clean, _, tempered_noise = outputs[1]
    torch.testing.assert_close(tempered_state, state)
    torch.testing.assert_close(tempered_noise, noise)
    torch.testing.assert_close(tempered_clean, GaussianScoreTemperature(.9)(state, clean, path))
    assert not torch.equal(tempered_clean, clean)

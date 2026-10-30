"""Paper presets must assemble real training/generation components, not just parse."""

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from core.sweeps.configuration import compose
from core.training.module import TrainingModule

RECIPES = (
    'mdlm md4 radd udlm dfm_masked dfm_uniform gidd_elbo cdcd langflow '
    'spherical_vmf cadd bd3lm llada_style ardm_style eb_masked flexmdm_style '
).split()


@pytest.mark.parametrize('recipe', RECIPES)
@pytest.mark.parametrize('task', ['sudoku', 'text8', 'tinystories'])
def test_recipe_task_composition(recipe, task):
    config = compose(recipe, [f'task={task}'])
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    assert config.task._target_.startswith('core.tasks.')


@pytest.mark.parametrize('recipe', RECIPES)
def test_recipe_loss_gradients_and_generation(recipe):
    torch.manual_seed(7)
    config = compose(recipe)
    backbone = config.model.get('backbone', config.model)
    backbone.hidden_size = 16
    backbone.n_blocks = 1
    backbone.n_heads = 2
    backbone.cond_dim = 8
    backbone.dropout = 0.0
    config.training.ema = 0
    config.task.length = 4
    if 'steps' in config.sampler:
        config.sampler.steps = 2
    if 'max_steps' in config.sampler:
        config.sampler.max_steps = 3
    if recipe == 'flexmdm_style':
        config.sampler.max_length = 16
        config.sampler.overflow = 'truncate'
    if recipe == 'bd3lm':
        config.block_diffusion.block_size = 2
    module = TrainingModule(config)
    tokens = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, 0] = False
    result = module.algorithm.loss(tokens, mask, training=True)
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    module.eval()
    with torch.no_grad():
        generated, info = module.algorithm.generate(instantiate(config.sampler), tokens, ~mask)
    assert generated.shape[0] == tokens.shape[0]
    assert torch.equal(generated[:, 0], tokens[:, 0])
    assert info['nfe'] > 0

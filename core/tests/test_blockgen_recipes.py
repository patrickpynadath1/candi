"""Fixed-block paper presets compose independently of tasks and execute both paths."""

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from core.sweeps.configuration import compose
from core.training.module import TrainingModule


@pytest.mark.parametrize('recipe', ['blockgen_masked', 'blockgen_uniform',
                                    'blockgen_masked_mixture', 'blockgen_uniform_mixture'])
@pytest.mark.parametrize('task', ['sudoku', 'text8', 'tinystories'])
def test_blockgen_task_composition(recipe, task):
    cfg = compose(recipe, [f'task={task}'])
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert cfg.block_diffusion.enabled and cfg.block_diffusion.x0_causal
    assert cfg.block_diffusion.clean_loss == 'none'
    assert cfg.model.adaLN and not cfg.algorithm.time_conditioning
    assert cfg.sampler.steps == 15
    assert cfg.model.n_blocks == 12 and cfg.model.hidden_size == 768


@pytest.mark.parametrize('recipe', ['blockgen_masked', 'blockgen_uniform'])
@pytest.mark.parametrize('block_size', [2, 4])
def test_blockgen_trains_and_generates_with_cache_parity(recipe, block_size):
    torch.manual_seed(19)
    cfg = compose(recipe, {
        'model.hidden_size': 16, 'model.n_heads': 2, 'model.n_blocks': 1,
        'model.cond_dim': 8, 'model.dropout': 0.0, 'task.length': 8,
        'block_diffusion.block_size': block_size, 'sampler.steps': 2,
        'training.ema': 0,
    })
    algorithm = TrainingModule(cfg).algorithm
    # Avoid the initial zero output projection hiding attention/cache errors.
    torch.nn.init.normal_(algorithm.model.output_layer.linear.weight, std=0.02)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, :block_size] = False
    result = algorithm.loss(tokens, mask, training=True)
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    grads = [p.grad for p in algorithm.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    sampler = instantiate(cfg.sampler)
    torch.manual_seed(20)
    cached, info = algorithm.generate(sampler, tokens, ~mask)
    algorithm.cache_clean_prefix = False
    torch.manual_seed(20)
    uncached, uncached_info = algorithm.generate(sampler, tokens, ~mask)
    assert torch.equal(cached, uncached)
    assert torch.equal(cached[~mask], tokens[~mask])
    assert (cached >= 0).all() and (cached < cfg.task.vocab_size).all()
    assert info['nfe'] == uncached_info['nfe'] == (8 // block_size - 1) * 3

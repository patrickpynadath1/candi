import pytest
import torch
from hydra.utils import instantiate
from core.sweeps.configuration import compose


@pytest.mark.parametrize('family', ['blurring', 'gaussian'])
def test_loss_sampler_observes_training_but_not_validation(family):
    cfg = compose(f'tinygsm_{family}_sc_adaptive')
    cfg.task.vocab_size = 11
    cfg.task.length = 7
    cfg.model.hidden_size = 16
    cfg.model.cond_dim = 8
    cfg.model.n_blocks = 1
    cfg.model.n_heads = 2
    cfg.model.dropout = 0
    cfg.time_sampler.warmup = 0
    cfg.time_sampler.refit_every = 1
    algorithm = instantiate(cfg.algorithm)
    tokens = torch.randint(0, 11, (3, 7))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    loss = algorithm.loss(tokens, mask, training=True, step=1)['loss']
    loss.backward()
    assert torch.isfinite(loss)
    assert algorithm.time_sampler.count.item() == 3
    algorithm.loss(tokens, mask, training=False, step=2)
    assert algorithm.time_sampler.count.item() == 3

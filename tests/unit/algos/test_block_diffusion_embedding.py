from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

import main  # registers Hydra resolvers
from algos import algo_registry
from algos.block_diffusion import sample_monotone_times
from algos.rank_schedules import SphereRankSchedule
from dataloader import Text8Tokenizer
from samplers import create_sampler


def config_for(schedule='linear', adaptive=False):
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[3] / 'configs'), version_base=None):
        return compose(config_name='config', overrides=[
            '+experiment=tinygsm_block_embedding',
            f'algo.interpolation_schedule={schedule}', f'noise.adaptive={str(adaptive).lower()}',
            'model.hidden_size=16', 'model.cond_dim=16', 'model.n_heads=2',
            'model.n_blocks=1', 'model.length=12', 'model.dropout=0',
            'algo.block_size=4', 'algo.rolling_warmup_steps=2',
            'algo.noise_curriculum_steps=0', 'training.ema=0', 'wandb=null',
        ])


@pytest.mark.parametrize('schedule', ['linear', 'linear_rank', 'geometric_rank'])
@pytest.mark.parametrize('adaptive', [False, True])
def test_monotone_training_backward_refit_and_both_rolling_samplers(schedule, adaptive, tmp_path):
    cfg = config_for(schedule, adaptive)
    cfg.algo.logit_kl_weight = 0.1
    cfg.noise.adaptive_buffer_size = 8
    cfg.noise.adaptive_refit_every = 1
    cfg.noise.adaptive_warmup_steps = 0
    cfg.noise.adaptive_n_knots = 3
    cfg.noise.adaptive_snapshot_dir = str(tmp_path)
    model = algo_registry.get(cfg.algo.name)(cfg, Text8Tokenizer())
    times = sample_monotone_times(8, 8, 4)
    u = model.interpolation(times).reshape(8, 2, 4)
    assert (u.diff(dim=-1) >= 0).all()
    assert torch.allclose(model.interpolation(torch.tensor([0., 1.])), torch.tensor([0., 1.]))
    tokens = torch.randint(0, 20, (8, 12))
    loss = model._loss(tokens, torch.ones_like(tokens), train_mode=True).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters())
    if adaptive:
        assert model.noise.refit_count >= 1
        u = model.interpolation(times).reshape(8, 2, 4)
        assert (u.diff(dim=-1) >= 0).all()
        assert torch.allclose(model.interpolation(torch.tensor([0., 1.])), torch.tensor([0., 1.]))
        restored = algo_registry.get(cfg.algo.name)(cfg, Text8Tokenizer())
        restored.load_state_dict(model.state_dict())
        assert torch.allclose(restored.interpolation(times), model.interpolation(times))
    for predictor in ('block_diffusion', 'block_diffusion_euler'):
        cfg.sampling.predictor = predictor
        create_sampler(cfg).validate_model(model)
        result = model.eval().generate_samples(2, num_steps=8)
        assert result.shape == (2, 12)
        assert (result >= 0).all() and (result < len(Text8Tokenizer())).all()


@pytest.mark.parametrize('schedule', ['linear_rank', 'geometric_rank'])
def test_corruption_conditioning_and_transport_share_candi_rank_mapping(schedule):
    cfg = config_for(schedule)
    model = algo_registry.get(cfg.algo.name)(cfg, Text8Tokenizer()).eval()
    times = torch.tensor([[0., .2, .6, 1.]])
    expected = SphereRankSchedule(16, .01, .49, .01)(times, coupling='linear' if schedule == 'linear_rank' else 'geometric')
    assert torch.allclose(model.interpolation(times), expected)
    tokens = torch.tensor([[1, 2, 3, 4]])
    clean = model._normalized_embeddings(tokens)
    noise = torch.randn_like(clean)
    observed = []
    original = model.backbone.forward_masked_embeddings
    def capture(*args, **kwargs):
        observed.append(kwargs['noise_times'].clone())
        return original(*args, **kwargs)
    model.backbone.forward_masked_embeddings = capture
    _, _, latent, _ = model.doubled_logits(tokens, times, noise)
    assert torch.allclose(latent, (1-expected[..., None])*clean + expected[..., None]*noise)
    assert torch.allclose(observed[-1][:, :4], expected)
    model.active_window_logits(tokens[:, :1], latent, times)
    assert torch.allclose(observed[-1][:, :4], expected)
    next_times = times * .5
    next_u = model.interpolation(next_times)
    for method in ('euler', 'renoise'):
        updated = model.update_rolling_latents(latent, clean, times, next_times, method=method, noise_correlation=1.)
        assert torch.allclose(updated, (1-next_u[..., None])*clean + next_u[..., None]*noise, atol=1e-6)


def test_rejects_truncated_rank_endpoints_for_rolling_prior():
    cfg = config_for('linear_rank')
    cfg.algo.rank_endpoint_width = 0
    with pytest.raises(ValueError, match='rank_endpoint_width'):
        algo_registry.get(cfg.algo.name)(cfg, Text8Tokenizer())


def test_adaptive_observations_use_valid_position_times_and_raw_noisy_loss():
    cfg = config_for('linear_rank', True)
    model = algo_registry.get(cfg.algo.name)(cfg, Text8Tokenizer())
    observed = []
    model.noise.record_time_loss_pair = lambda t, loss, step: observed.append((t, loss))
    times = torch.tensor([[.1, .3], [.4, .8]])
    losses = torch.tensor([[1., 3.], [4., 8.]])
    valid = torch.tensor([[0, 1], [1, 0]])
    model._record_block_time_loss(times, losses, valid, True)
    assert torch.equal(observed[0][0], torch.tensor([.3, .4]))
    assert torch.equal(observed[0][1], torch.tensor([3., 4.]))
    model._record_block_time_loss(times, losses, valid, False)
    assert len(observed) == 1


def test_linear_variant_matches_original_corruption_and_logits():
    cfg = config_for('linear')
    model = algo_registry.get(cfg.algo.name)(cfg, Text8Tokenizer()).eval()
    cfg.algo.name = 'block_diffusion'
    original = algo_registry.get(cfg.algo.name)(cfg, Text8Tokenizer()).eval()
    original.load_state_dict(model.state_dict())
    tokens = torch.tensor([[1, 2, 3, 4]])
    times = torch.tensor([[0., .25, .5, 1.]])
    noise = torch.randn(1, 4, 16)
    for actual, expected in zip(model.doubled_logits(tokens, times, noise), original.doubled_logits(tokens, times, noise)):
        assert torch.allclose(actual, expected, atol=1e-6)

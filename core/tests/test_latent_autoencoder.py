"""Latent training semantics, gradient ownership, visibility and generation."""

import pytest
import torch
from torch import nn

from core.sweeps.configuration import compose
from core.training.module import TrainingModule
from core.samplers.latent_autoencoder import LatentAutoencoderSampler


def build(detach=True):
    cfg = compose(
        "latent_autoencoder",
        [
            "latent.hidden_size=32",
            "latent.n_heads=4",
            "latent.encoder_layers=2",
            "latent.denoiser_layers=2",
            "latent.decoder_layers=2",
            "latent.block_size=2",
            "latent.dropout=0",
            "task.length=8",
            "training.ema=0",
            "training.performance.compile.enabled=false",
            f"algorithm.detach_encoder_for_denoising={str(detach).lower()}",
        ],
    )
    module = TrainingModule(cfg)
    # Open deliberately zero-initialized heads and adaLN gates for gradient checks.
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.05)
    return module


class FixedTime:
    def __init__(self):
        self.observation = None

    def sample(self, count, device, **kwargs):
        return torch.full((count,), 0.5, device=device)

    def observe(self, time, values, mask, step):
        assert not values.requires_grad
        self.observation = values.clone()


def test_exact_losses_and_cdcd_observes_decoded_ce_not_mse(monkeypatch):
    algo = build().algorithm
    sampler = FixedTime()
    # nn.Module's sampler slot is registered, so use a recording hook instead.
    monkeypatch.setattr(algo.time_sampler, "sample", sampler.sample)
    monkeypatch.setattr(algo.time_sampler, "observe", sampler.observe)
    x = torch.randint(0, 9, (2, 8))
    mask = torch.ones_like(x, dtype=torch.bool)
    recorded = []
    decode = algo.model.decode

    def record(*args, **kwargs):
        value = decode(*args, **kwargs)
        recorded.append(value)
        return value

    monkeypatch.setattr(algo.model, "decode", record)
    result = algo.loss(x, mask, training=True)
    expected = (
        -recorded[-1].float().log_softmax(-1).gather(-1, x[..., None]).squeeze(-1)
    )
    torch.testing.assert_close(sampler.observation, expected)
    torch.testing.assert_close(
        result["loss"], result["latent_mse"] + result["clean_decoding_loss"]
    )
    assert not result["decoded_denoising_ce"].requires_grad
    assert algo.time_sampler.pending_count == 0  # recording hook replaced the fitter


@pytest.mark.parametrize("detach", [True, False])
def test_gradient_routing(detach):
    algo = build(detach).algorithm
    x = torch.randint(0, 9, (2, 8))
    mask = torch.ones_like(x, dtype=torch.bool)
    result = algo.loss(x, mask)
    result["latent_mse"].backward()
    encoder_grad = sum(
        float(p.grad.abs().sum())
        for p in algo.model.encoder.parameters()
        if p.grad is not None
    )
    assert (encoder_grad == 0) if detach else (encoder_grad > 0)
    assert all(p.grad is None for p in algo.model.decoder.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in algo.model.denoiser.parameters()
    )
    algo.zero_grad(set_to_none=True)
    algo.loss(x, mask)["clean_decoding_loss"].backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in algo.model.encoder.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in algo.model.decoder.parameters()
    )
    assert all(p.grad is None for p in algo.model.denoiser.parameters())


@torch.no_grad()
def test_normalization_visibility_and_encoder_cache():
    model = build().algorithm.model.eval()
    x = torch.randint(0, 9, (2, 8))
    z = model.encode(x)
    torch.testing.assert_close(z.norm(dim=-1), torch.full((2, 8), 32**0.5))
    altered = x.clone()
    altered[:, 2:] = (altered[:, 2:] + 1) % 9
    changed = model.encode(altered)
    torch.testing.assert_close(z[:, :2], changed[:, :2])
    # A fixed noisy block cannot read its clean target or future blocks.
    state = torch.randn_like(z)
    time = torch.ones(2) * 0.5
    base = model.denoise(state, time, z)
    torch.testing.assert_close(base[:, :4], model.denoise(state, time, changed)[:, :4])
    torch.testing.assert_close(
        model.decode(state, z)[:, :4], model.decode(state, changed)[:, :4]
    )
    cache = {}
    for end in (0, 2, 4, 8):
        actual = model.encode(x[:, :end], cache=cache)
        torch.testing.assert_close(actual, z[:, :end], atol=2e-6, rtol=1e-5)
    changed = x.clone()
    changed[:, 0] = (changed[:, 0] + 1) % 9
    torch.testing.assert_close(
        model.encode(changed, cache=cache), model.encode(changed)
    )


def test_cdcd_updates_and_partial_clean_blocks_rejected():
    algo = build().algorithm
    x = torch.randint(0, 9, (2, 8))
    mask = torch.ones_like(x, dtype=torch.bool)
    algo.loss(x, mask, training=True, step=0)
    assert algo.time_sampler.pending_count > 0
    algo.time_sampler.post_optimizer_step()
    assert algo.time_sampler.updates > 0
    bad = mask.clone()
    bad[:, 0] = False
    with pytest.raises(ValueError, match="whole blocks"):
        algo.loss(x, mask, corruption_mask=bad)


@torch.no_grad()
def test_generation_preserves_prompt_counts_calls_and_resets_caches():
    algo = build().algorithm.eval()
    x = torch.randint(0, 9, (2, 8))
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, :2] = True
    sampler = LatentAutoencoderSampler(steps=2)
    for _ in range(2):
        samples, metadata = algo.generate(sampler, x, mask)
        assert samples.shape == x.shape
        assert (
            samples.min() >= 0
            and samples.max() < algo.model.decoder.output_layer.linear.out_features
        )
        torch.testing.assert_close(samples[:, :2], x[:, :2])
        assert metadata["nfe_per_sample"] == [12, 12]
    mask[:, 2] = True
    with pytest.raises(ValueError, match="block boundary"):
        algo.generate(sampler, x, mask)


def test_checkpoint_roundtrip():
    first = build()
    second = build()
    second.load_state_dict(first.state_dict())
    x = torch.randint(0, 9, (2, 8))
    mask = torch.ones_like(x, dtype=torch.bool)
    torch.manual_seed(3)
    a = first.algorithm.loss(x, mask)["loss"]
    torch.manual_seed(3)
    b = second.algorithm.loss(x, mask)["loss"]
    torch.testing.assert_close(a, b)

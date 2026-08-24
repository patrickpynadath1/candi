from types import SimpleNamespace

import torch

from algos.block_diffusion import (
    BlockDiffusion,
    block_diffusion_mask,
    detached_clean_logit_kl,
    delayed_linear_curriculum_cap,
    flow_interpolate,
    rolling_latent_update,
    sample_monotone_times,
    sample_position_times,
)
from models.dit import DIT


def _tiny_causal_dit(*, anchored_adaln=False):
    config = SimpleNamespace(
        algo=SimpleNamespace(
            causal_attention=True,
            time_conditioning=anchored_adaln,
            adaLN=anchored_adaln,
        ),
        model=SimpleNamespace(
            hidden_size=16,
            cond_dim=16,
            n_heads=2,
            n_blocks=2,
            dropout=0.0,
            scale_by_sigma=False,
            tie_word_embeddings=False,
        ),
    )
    return DIT(config, vocab_size=13).eval()


def test_block_mask_has_clean_causal_and_noisy_block_causal_edges():
    mask = block_diffusion_mask(length=4, block_size=2)
    expected_noisy_to_noisy = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=torch.bool,
    )
    expected_noisy_to_clean = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask[:4, :4], expected_noisy_to_noisy)
    assert torch.equal(mask[:4, 4:], expected_noisy_to_clean)
    assert torch.equal(mask[4:, 4:], torch.ones(4, 4, dtype=torch.bool).tril())
    assert not mask[4:, :4].any()


def test_sampled_position_times_are_monotone_and_end_at_block_maximum():
    times = sample_monotone_times(128, 10, 4, max_time=0.7)
    assert times.shape == (128, 10)
    assert times.min() >= 0
    assert times.max() <= 0.7
    for start, stop in ((0, 4), (4, 8), (8, 10)):
        assert torch.all(times[:, start + 1 : stop] >= times[:, start : stop - 1])


def test_sequence_constant_times_share_one_noise_level_everywhere():
    times = sample_position_times(
        64, 17, 4, mode="sequence_constant", max_time=0.6
    )
    assert times.shape == (64, 17)
    assert times.min() >= 0 and times.max() <= 0.6
    assert torch.equal(times, times[:, :1].expand_as(times))


def test_linear_flow_has_exact_clean_and_noise_endpoints():
    clean = torch.randn(2, 3, 5)
    noise = torch.randn_like(clean)
    times = torch.tensor([[0.0, 0.5, 1.0]]).expand(2, -1)
    latent, returned_noise = flow_interpolate(clean, times, noise)
    assert torch.equal(returned_noise, noise)
    assert torch.allclose(latent[:, 0], clean[:, 0])
    assert torch.allclose(latent[:, 2], noise[:, 2])


def test_delayed_noise_curriculum_has_ar_ramp_and_full_noise_phases():
    values = [
        delayed_linear_curriculum_cap(step, 2500, 5000, 1.0)
        for step in (0, 2499, 2500, 3750, 5000, 9000)
    ]
    assert values == [0.0, 0.0, 0.0, 0.5, 1.0, 1.0]


@torch.no_grad()
def test_forward_and_reverse_logit_kl_are_zero_for_equal_logits():
    logits = torch.randn(2, 3, 7)
    for direction in ("forward", "reverse"):
        kl = detached_clean_logit_kl(
            logits, logits, direction=direction, temperature=2.0
        )
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_logit_kl_detaches_clean_teacher_for_both_directions():
    for direction in ("forward", "reverse"):
        noisy = torch.randn(2, 3, 7, requires_grad=True)
        clean = torch.randn(2, 3, 7, requires_grad=True)
        loss = detached_clean_logit_kl(
            noisy, clean, direction=direction, temperature=1.5
        ).mean()
        noisy_gradient, clean_gradient = torch.autograd.grad(
            loss, (noisy, clean), allow_unused=True
        )
        assert noisy_gradient is not None and noisy_gradient.abs().sum() > 0
        assert clean_gradient is None


def test_zero_noise_doubled_mask_is_exactly_ar_equivalent():
    model = _tiny_causal_dit()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    clean = torch.nn.functional.normalize(model.get_embedding(tokens), dim=-1)
    doubled = torch.cat((clean, clean), dim=1)
    positions = torch.arange(tokens.shape[1]).repeat(2).unsqueeze(0)
    logits = model.forward_masked_embeddings(
        doubled,
        block_diffusion_mask(tokens.shape[1], 3),
        positions,
    )
    noisy_logits, clean_logits = logits.chunk(2, dim=1)
    assert torch.allclose(noisy_logits, clean_logits, atol=1e-6, rtol=1e-6)


def test_anchored_tokenwise_adaln_is_exactly_plain_ar_at_zero_noise():
    torch.manual_seed(7)
    plain = _tiny_causal_dit()
    for parameter in plain.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    conditioned = _tiny_causal_dit(anchored_adaln=True)
    conditioned.load_state_dict(plain.state_dict(), strict=False)
    for name, parameter in conditioned.named_parameters():
        if "adaLN_modulation" in name or "sigma_map" in name:
            torch.nn.init.normal_(parameter, std=0.1)

    tokens = torch.tensor([[1, 2, 3, 4]])
    embeddings = torch.nn.functional.normalize(plain.get_embedding(tokens), dim=-1)
    positions = torch.arange(tokens.shape[1]).unsqueeze(0)
    mask = torch.ones(4, 4, dtype=torch.bool).tril()
    plain_logits = plain.forward_masked_embeddings(embeddings, mask, positions)
    zero_logits = conditioned.forward_masked_embeddings(
        embeddings, mask, positions, noise_times=torch.zeros_like(tokens).float()
    )
    assert torch.equal(plain_logits, zero_logits)

    nonzero_logits = conditioned.forward_masked_embeddings(
        embeddings,
        mask,
        positions,
        noise_times=torch.tensor([[0.0, 0.25, 0.5, 1.0]]),
    )
    assert not torch.allclose(nonzero_logits, plain_logits)


def test_euler_update_exactly_recomposes_linear_flow():
    x0 = torch.randn(2, 3, 5)
    eps = torch.randn_like(x0)
    old = torch.full((2, 3), 0.8)
    new = torch.full((2, 3), 0.3)
    latent, _ = flow_interpolate(x0, old, eps)
    updated = rolling_latent_update(
        latent, x0, old, new, method="euler"
    )
    expected, _ = flow_interpolate(x0, new, eps)
    assert torch.allclose(updated, expected, atol=1e-6, rtol=1e-6)


def test_renoise_update_obeys_requested_endpoint():
    x0 = torch.randn(2, 3, 5)
    latent = torch.randn_like(x0)
    old = torch.full((2, 3), 0.8)
    new = torch.zeros(2, 3)
    updated = rolling_latent_update(
        latent, x0, old, new, method="renoise", noise_correlation=0.5
    )
    assert torch.equal(updated, x0)


def test_logits_to_x0_accepts_bfloat16_logits_and_returns_stable_fp32():
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vocab_embed = torch.nn.Module()
            self.vocab_embed.embedding = torch.nn.Parameter(torch.randn(7, 5))

    model = object.__new__(BlockDiffusion)
    torch.nn.Module.__init__(model)
    model.backbone = Backbone()
    model.embedding_scale = 1.0
    result = model.logits_to_x0(torch.randn(2, 3, 7, dtype=torch.bfloat16))
    assert result.dtype == torch.float32
    assert result.shape == (2, 3, 5)
    assert torch.isfinite(result).all()

    argmax_result = model.logits_to_x0(
        torch.randn(2, 3, 7, dtype=torch.bfloat16), projection="argmax"
    )
    assert argmax_result.dtype == torch.float32
    assert argmax_result.shape == (2, 3, 5)
    assert torch.allclose(argmax_result.norm(dim=-1), torch.ones(2, 3))

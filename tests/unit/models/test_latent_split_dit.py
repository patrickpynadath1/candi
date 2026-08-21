from types import SimpleNamespace

import pytest
import torch

from models.latent_split_dit import LatentSplitDIT


def _config(ratio=0.5, blocks=4):
    return SimpleNamespace(
        model=SimpleNamespace(
            hidden_size=8,
            cond_dim=8,
            n_blocks=blocks,
            n_heads=2,
            dropout=0.0,
            encoder_ratio=ratio,
            tie_word_embeddings=False,
        )
    )


def test_split_and_latent_shapes():
    model = LatentSplitDIT(_config(ratio=0.25), vocab_size=6)
    tokens = torch.tensor([[0, 1, 2], [3, 4, 5]])
    latent = model.encode_tokens(tokens)
    logits = model.decode_latents(latent, torch.tensor([0.2, 0.8]))

    assert model.encoder_blocks == 1
    assert model.decoder_blocks == 3
    assert latent.shape == (2, 3, 8)
    assert logits.shape == (2, 3, 6)
    assert torch.allclose(latent.mean(-1), torch.zeros(2, 3), atol=1e-5)
    assert torch.allclose(
        latent.var(-1, unbiased=False), torch.ones(2, 3), atol=2e-4
    )
    assert not tuple(model.latent_norm.parameters())


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.1])
def test_invalid_split_ratio_is_rejected(ratio):
    with pytest.raises(ValueError, match="encoder_ratio"):
        LatentSplitDIT(_config(ratio=ratio), vocab_size=6)


def test_feedback_modes_return_embeddings(monkeypatch):
    model = LatentSplitDIT(_config(), vocab_size=3)
    logits = torch.tensor([[[0.0, 1.0, 3.0]]])
    greedy = model.feedback_embeddings(logits, "greedy")
    soft = model.feedback_embeddings(logits, "soft")
    monkeypatch.setattr(torch, "multinomial", lambda probabilities, _: probabilities.argmax(-1, keepdim=True))
    sampled = model.feedback_embeddings(logits, "sample")

    assert torch.equal(greedy, model.embed_tokens(torch.tensor([[2]])))
    assert torch.equal(sampled, greedy)
    assert soft.shape == greedy.shape
    assert not torch.equal(soft, greedy)


def test_variational_posterior_shapes_and_reparameterization(monkeypatch):
    model = LatentSplitDIT(_config(), vocab_size=5)
    tokens = torch.tensor([[0, 1, 2]])
    mean, log_variance = model.posterior_from_tokens(tokens)
    monkeypatch.setattr(torch, "randn_like", lambda tensor: torch.ones_like(tensor))
    sample = model.reparameterize(mean, log_variance)

    assert mean.shape == log_variance.shape == (1, 3, 8)
    assert torch.equal(log_variance, torch.zeros_like(log_variance))
    assert torch.allclose(sample, mean + 1)


def test_encoder_and_decoder_receive_gradients():
    torch.manual_seed(4)
    model = LatentSplitDIT(_config(), vocab_size=5)
    # The standard zero-init head delays backbone gradients for one update.
    torch.nn.init.normal_(model.output_layer.linear.weight, std=0.02)
    # Open the zero-initialized AdaLN residual gates for this structural test.
    with torch.no_grad():
        for block in model.blocks[model.encoder_blocks :]:
            dim = model.config.model.hidden_size
            block.adaLN_modulation.bias[2 * dim : 3 * dim].fill_(1)
            block.adaLN_modulation.bias[5 * dim : 6 * dim].fill_(1)
    tokens = torch.tensor([[0, 1, 2], [2, 3, 4]])
    logits = model.decode_latents(
        model.encode_tokens(tokens), torch.tensor([0.25, 0.75])
    )
    torch.nn.functional.cross_entropy(logits.transpose(1, 2), tokens).backward()

    assert model.blocks[0].attn_qkv.weight.grad is not None
    assert model.blocks[-1].attn_qkv.weight.grad is not None
    assert torch.count_nonzero(model.blocks[0].attn_qkv.weight.grad) > 0
    assert torch.count_nonzero(model.blocks[-1].attn_qkv.weight.grad) > 0

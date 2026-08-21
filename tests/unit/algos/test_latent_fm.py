import torch

from algos.latent_fm import LatentFlowMatching


def _bare(**attrs):
    model = object.__new__(LatentFlowMatching)
    torch.nn.Module.__init__(model)
    for key, value in attrs.items():
        setattr(model, key, value)
    return model


def test_interpolation_endpoints():
    noise = torch.randn(2, 3, 4)
    clean = torch.randn(2, 3, 4)
    assert torch.equal(
        LatentFlowMatching.interpolate(noise, clean, torch.zeros(2)), noise
    )
    assert torch.equal(
        LatentFlowMatching.interpolate(noise, clean, torch.ones(2)), clean
    )


def test_posterior_kl_matches_standard_normal():
    mean = torch.zeros(2, 3, 4)
    log_variance = torch.zeros_like(mean)
    shifted = mean + 1

    assert torch.allclose(
        LatentFlowMatching.posterior_kl(mean, log_variance), torch.tensor(0.0)
    )
    assert torch.allclose(
        LatentFlowMatching.posterior_kl(shifted, log_variance), torch.tensor(6.0)
    )
    assert torch.allclose(
        LatentFlowMatching.posterior_kl_per_scalar(shifted, log_variance),
        torch.tensor(0.5),
    )


def test_masked_ce_mean_excludes_prompt_positions():
    values = torch.tensor([[9.0, 9.0, 2.0, 4.0]])
    valid = torch.tensor([[0, 0, 1, 1]])

    assert LatentFlowMatching.masked_mean(values, valid) == 3
    assert LatentFlowMatching.masked_mean(values) == 6


def test_flow_map_reaches_reencoded_prediction_and_preserves_prompt_input():
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def decode_latents(self, latent, _time):
            return torch.zeros(*latent.shape[:2], 3)

        def feedback_embeddings(self, logits, _mode, _temperature):
            return torch.zeros(*logits.shape[:2], 4)

        def embed_tokens(self, tokens):
            return torch.nn.functional.one_hot(tokens, 4).float()

        def encode_embeddings(self, embeddings):
            self.seen = embeddings
            return embeddings + 2

    backbone = Backbone()
    model = _bare(
        backbone=backbone,
        feedback_mode="soft",
        feedback_temperature=1.0,
    )
    latent = torch.randn(1, 2, 4)
    prompt = torch.tensor([[1, 0]])
    mask = torch.tensor([[1, 0]])
    result, _ = model.sample_latent_flow_map(
        latent,
        torch.tensor([0.5]),
        torch.tensor([1.0]),
        prompt_tokens=prompt,
        prompt_mask=mask,
    )

    assert torch.equal(backbone.seen[:, 0], backbone.embed_tokens(prompt)[:, 0])
    assert torch.allclose(result, backbone.seen + 2)

from types import SimpleNamespace

import torch

from samplers.base import SamplingRequest
from samplers.replaid_keep_ddpm import RePlaidKeepDDPMSampler


def test_reveal_probability_matches_conditional_mask_transition():
    gamma_t = torch.tensor([2.0, 0.0])
    gamma_s = torch.tensor([1.0, -1.0])
    keep_t = torch.sigmoid(-gamma_t)
    keep_s = torch.sigmoid(-gamma_s)

    actual = RePlaidKeepDDPMSampler.reveal_probability(gamma_t, gamma_s)

    assert torch.allclose(actual, (keep_s - keep_t) / (1 - keep_t))


def test_progressive_reveal_preserves_prompt_and_passes_hard_clean_context():
    class Model:
        device = torch.device("cpu")
        num_tokens = 4
        vocab_size = 5
        config = SimpleNamespace(
            sampling=SimpleNamespace(steps=3),
            algo=SimpleNamespace(reveal_strategy="argmax"),
        )

        def __init__(self):
            self.calls = []

        def prior_sample(self, count, length):
            return torch.zeros(count, length, 2)

        def sampling_embeddings(self, tokens):
            return torch.stack((tokens.float(), -tokens.float()), -1)

        def sampling_gamma(self, t):
            return 6 * t - 3

        def sample_replaid_keep_denoise(self, z_t, t, kept_tokens, kept_mask):
            del z_t
            self.calls.append((kept_tokens.clone(), kept_mask.clone()))
            logits = torch.full((*kept_tokens.shape, self.vocab_size), -20.0)
            logits[..., 3] = 0
            predicted = self.sampling_embeddings(torch.full_like(kept_tokens, 3))
            return torch.log_softmax(logits, -1), predicted, self.sampling_gamma(t)

    torch.manual_seed(4)
    model = Model()
    prompt_tokens = torch.tensor([[1, 2, 0, 0]])
    prompt_mask = torch.tensor([[True, True, False, False]])
    result = RePlaidKeepDDPMSampler().sample(
        model,
        SamplingRequest(
            num_samples=1,
            num_steps=3,
            eps=0.0,
            prompt_tokens=prompt_tokens,
            prompt_mask=prompt_mask,
        ),
    )

    assert torch.equal(result.samples[0, :2], prompt_tokens[0, :2])
    assert torch.equal(result.samples[0, 2:], torch.tensor([3, 3]))
    assert all(torch.all(mask[:, :2]) for _, mask in model.calls)
    counts = [mask.sum().item() for _, mask in model.calls]
    assert counts == sorted(counts)
    for tokens, mask in model.calls:
        assert torch.equal(tokens[mask][:2], prompt_tokens[prompt_mask][:2])

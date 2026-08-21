from types import SimpleNamespace

import torch

from samplers.base import SamplingRequest
from samplers.replaid_keep_entropy_bounded import RePlaidKeepEntropyBoundedSampler


def test_entropy_bounded_replaid_preserves_prompt_and_keeps_latent_rejections():
    class Model:
        device = torch.device("cpu")
        dtype = torch.float32
        num_tokens = 4
        vocab_size = 5
        config = SimpleNamespace(sampling=SimpleNamespace(steps=4))

        def __init__(self):
            self.calls = []

        def prior_sample(self, count, length):
            return torch.zeros(count, length, 2)

        def sampling_embeddings(self, tokens):
            return torch.stack((tokens.float(), -tokens.float()), -1)

        def sampling_gamma(self, t):
            return 6 * t - 3

        def sample_replaid_keep_denoise(self, z_t, t, kept_tokens, kept_mask):
            self.calls.append((z_t.clone(), kept_tokens.clone(), kept_mask.clone()))
            logits = torch.full((*kept_tokens.shape, self.vocab_size), -9.0)
            # Distinct entropy ranks ensure gamma=0 accepts one active token
            # per evaluation, while every categorical mode remains token 3.
            for position in range(self.num_tokens):
                logits[:, position, 3] = 4.0 - position
                logits[:, position, 2] = position * 0.5
            predicted = self.sampling_embeddings(torch.full_like(kept_tokens, 3))
            return torch.log_softmax(logits, -1), predicted, self.sampling_gamma(t)

    torch.manual_seed(7)
    model = Model()
    prompt_tokens = torch.tensor([[1, 0, 0, 0]])
    prompt_mask = torch.tensor([[True, False, False, False]])
    result = RePlaidKeepEntropyBoundedSampler(0.0).sample(
        model,
        SamplingRequest(
            num_samples=1,
            num_steps=4,
            eps=0.0,
            prompt_tokens=prompt_tokens,
            prompt_mask=prompt_mask,
        ),
    )

    assert result.samples[0, 0] == 1
    assert torch.equal(result.samples[0, 1:], torch.tensor([3, 3, 3]))
    counts = [mask.sum().item() for _, _, mask in model.calls]
    assert counts == sorted(counts)
    assert result.metadata["nfe"] == 3
    for state, tokens, kept in model.calls:
        hard = model.sampling_embeddings(tokens)
        assert torch.allclose(state[kept], hard[kept])

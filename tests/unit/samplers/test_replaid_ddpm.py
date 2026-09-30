import torch
import pytest

from samplers.base import SamplingRequest
from samplers.replaid_ddpm import RePlaidDDPMSampler


def test_ddpm_transition_is_exact_at_zero_snr_gap():
    z_t = torch.randn(2, 3, 4)
    clean = torch.randn_like(z_t)
    gamma = torch.tensor([0.2, -0.3])

    result = RePlaidDDPMSampler.transition(
        z_t, clean, gamma, gamma, torch.randn_like(z_t)
    )

    assert torch.allclose(result, z_t)


def test_ddpm_transition_is_finite_for_a_reverse_step():
    z_t = torch.randn(2, 3, 4)
    clean = torch.randn_like(z_t)
    result = RePlaidDDPMSampler.transition(
        z_t,
        clean,
        torch.tensor([3.0, 3.0]),
        torch.tensor([1.0, 1.0]),
        torch.randn_like(z_t),
    )

    assert result.shape == z_t.shape
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("clean_context", [False, True])
def test_conditional_sampling_preserves_prompt_and_uses_one_forward_noise_path(clean_context):
    class Model:
        device = torch.device("cpu")
        num_tokens = 3
        vocab_size = 4
        config = type("Config", (), {"sampling": type("Sampling", (), {"steps": 2})(), "algo": {}})()

        def __init__(self):
            self.states = []

        def prior_sample(self, count, length):
            return torch.zeros(count, length, 2)

        def sampling_embeddings(self, tokens):
            return torch.nn.functional.one_hot(tokens % 2, 2).float()

        def sampling_gamma(self, t):
            return 4 * t - 2

        def sample_replaid_denoise(self, z_t, t, self_condition=None):
            del self_condition
            self.states.append(z_t.clone())
            logits = torch.full((*z_t.shape[:2], 4), -10.0)
            logits[..., 3] = 0
            clean = torch.zeros_like(z_t)
            return torch.log_softmax(logits, -1), clean, self.sampling_gamma(t)

    torch.manual_seed(4)
    model = Model()
    model.config.algo = {"clean_context": clean_context}
    prompt_tokens = torch.tensor([[1, 2, 0]])
    prompt_mask = torch.tensor([[True, True, False]])
    result = RePlaidDDPMSampler().sample(
        model,
        SamplingRequest(
            num_samples=1,
            num_steps=2,
            eps=0.1,
            prompt_tokens=prompt_tokens,
            prompt_mask=prompt_mask,
        ),
    )

    assert torch.equal(result.samples[0, :2], prompt_tokens[0, :2])
    embedding = model.sampling_embeddings(prompt_tokens)
    if clean_context:
        for state in model.states:
            assert torch.equal(state[prompt_mask], embedding[prompt_mask])
        return
    gamma_0 = model.sampling_gamma(torch.tensor([1.0]))
    gamma_1 = model.sampling_gamma(torch.tensor([0.55]))
    alpha_0, sigma_0 = torch.sigmoid(-gamma_0).sqrt(), torch.sigmoid(gamma_0).sqrt()
    alpha_1, sigma_1 = torch.sigmoid(-gamma_1).sqrt(), torch.sigmoid(gamma_1).sqrt()
    inferred_noise = (model.states[0] - alpha_0[:, None, None] * embedding) / sigma_0[:, None, None]
    expected_next = alpha_1[:, None, None] * embedding + sigma_1[:, None, None] * inferred_noise
    assert torch.allclose(
        model.states[1][prompt_mask], expected_next[prompt_mask], atol=1e-6
    )

from types import SimpleNamespace

import torch

from algos.spherical_utils import sphere_normalize
from samplers.base import SamplingRequest
from samplers.flm_euler import FLMEulerSampler, flm_euler_step
from samplers.sfm import sfm_compute_velocity


def test_flm_euler_step_matches_definition():
    z = torch.zeros(1, 2, 3)
    target = torch.ones_like(z) / 3
    out = flm_euler_step(z, target, torch.tensor([0.2]), torch.tensor([0.1]))
    assert torch.allclose(out, target / 8)


def test_exact_sfm_velocity_is_tangent():
    x = sphere_normalize(torch.randn(2, 3, 4))
    e = sphere_normalize(torch.randn(7, 4))
    logp = torch.randn(2, 3, 7).log_softmax(-1)
    velocity = sfm_compute_velocity(x, e, logp)
    assert torch.allclose((velocity * x).sum(-1), torch.zeros(2, 3), atol=2e-5)


class _FLM:
    sampling_family = "flm"
    sampling_capabilities = {"flm_prior", "flm_time_map", "flm_logits"}
    vocab_size = 5
    num_tokens = 4
    device = torch.device("cpu")
    config = SimpleNamespace(sampling=SimpleNamespace(steps=2))

    def prior_sample(self, batch, length):
        return torch.zeros(batch, length, self.vocab_size)

    def sampling_gamma(self, t):
        return t * 0.9

    def sampling_logits(self, x, t):
        return torch.zeros_like(x).scatter(
            -1, torch.full(x.shape[:-1] + (1,), 2, dtype=torch.long), 10
        )


def test_flm_sampler_preserves_prompt():
    model = _FLM()
    sampler = FLMEulerSampler()
    req = SamplingRequest(
        2,
        prompt_tokens=torch.tensor([[1, 0, 0, 0], [3, 0, 0, 0]]),
        prompt_mask=torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.bool),
    )
    result = sampler.sample(model, req)
    assert torch.equal(result.samples[:, 0], torch.tensor([1, 3]))
    assert torch.all(result.samples[:, 1:] == 2)

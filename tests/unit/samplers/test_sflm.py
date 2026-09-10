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


class _SFM:
    num_tokens = 2
    device = torch.device("cpu")
    config = SimpleNamespace(sampling=SimpleNamespace(steps=3))

    def __init__(self):
        self.states = []
        self.backbone = SimpleNamespace(
            get_sphere_embeddings=lambda t: self.sampling_embeddings()[t]
        )

    def prior_sample(self, batch, length):
        return torch.tensor([0.0, 0.0, 1.0]).expand(batch, length, 3).clone()

    def sampling_embeddings(self):
        return torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def sampling_alpha(self, t):
        return 1 - 0.999 * t

    def sampling_logits(self, x, t):
        self.states.append(x.clone())
        assert x.dtype == torch.float32
        return torch.tensor([0.9, 0.1]).log().expand(*x.shape[:2], 2)


def _sfm_config(**kwargs):
    defaults = dict(velocity="exact", noise_removal="greedy", use_float64=True)
    defaults.update(kwargs)
    return SimpleNamespace(
        sampling=SimpleNamespace(**defaults),
        algo=SimpleNamespace(slerp_precision="float64"),
    )


def test_sfm_topk_sampling_uses_probabilities_and_double_geometry(monkeypatch):
    import samplers.sfm as sfm

    original_draw, original_exp = sfm.sample_categorical, sfm.exp_map
    calls = []

    def draw(p):
        assert p.dtype == torch.float64
        assert torch.all(p >= 0)
        assert torch.allclose(p.sum(-1), torch.ones_like(p[..., 0]))
        result = original_draw(p)
        calls.append(result)
        return result

    def exp(x, delta, eps):
        assert x.dtype == delta.dtype == torch.float64
        return original_exp(x, delta, eps)

    monkeypatch.setattr(sfm, "sample_categorical", draw)
    monkeypatch.setattr(sfm, "exp_map", exp)
    torch.manual_seed(0)
    result = sfm.SFMSampler(_sfm_config(velocity="sample", top_k_velocity=99)).sample(
        _SFM(), SamplingRequest(10000, num_steps=2)
    )
    assert abs(calls[0].float().mean().item() - 0.1) < 0.01
    assert result.metadata["nfe"] == 2


def test_sfm_prompt_preservation_and_nfe():
    from samplers.sfm import SFMSampler

    model = _SFM()
    prompt = torch.tensor([[1, 0]])
    result = SFMSampler(_sfm_config()).sample(
        model,
        SamplingRequest(
            1, prompt_tokens=prompt, prompt_mask=torch.tensor([[True, False]])
        ),
    )
    assert len(model.states) == result.metadata["nfe"] == 3
    assert result.samples.tolist() == [[1, 0]]
    for state in model.states:
        assert torch.equal(state[0, 0], model.sampling_embeddings()[1])
        assert torch.allclose(state.norm(dim=-1), torch.ones(1, 2), atol=1e-6)


def test_sfm_filtering_applies_to_final_decoding():
    from samplers.sfm import SFMSampler

    for options in ({"top_k": 1}, {"p_nucleus": 0.8}):
        result = SFMSampler(_sfm_config(noise_removal="ancestral", **options)).sample(
            _SFM(), SamplingRequest(1000, num_steps=1)
        )
        assert torch.all(result.samples == 0)

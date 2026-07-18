from types import MethodType, SimpleNamespace

import pytest
import torch

from algos.absorbing_state import AbsorbingState
from algos.diffusion import Diffusion
from algos.uniform_state import UniformState


def _bare(cls, **attrs):
    instance = object.__new__(cls)
    torch.nn.Module.__init__(instance)
    instance._device = torch.device("cpu")
    instance._dtype = torch.float32
    for name, value in attrs.items():
        setattr(instance, name, value)
    return instance


def test_diffusion_process_model_input_is_identity():
    model = _bare(Diffusion)
    tokens = torch.tensor([[1, 2]])
    valid = torch.tensor([[1, 0]])
    assert model._process_model_input(tokens, valid) == (tokens, None, valid)


@pytest.mark.parametrize("time_conditioning", [True, False])
def test_process_sigma_reduces_sequence_dimension(time_conditioning):
    model = _bare(Diffusion, time_conditioning=time_conditioning)
    sigma = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    result = model._process_sigma(sigma)
    expected = torch.tensor([2.0, 3.0]) if time_conditioning else torch.zeros(2)
    assert torch.equal(result, expected)


def test_process_sigma_rejects_wrong_rank():
    model = _bare(Diffusion, time_conditioning=True)
    with pytest.raises(AssertionError):
        model._process_sigma(torch.ones(2))


def test_sample_t_is_bounded_and_antithetic_stratified():
    model = _bare(
        Diffusion,
        antithetic_sampling=True,
        sampling_eps=0.1,
        config=SimpleNamespace(loader=SimpleNamespace(global_batch_size=4)),
    )
    torch.manual_seed(4)
    values = model._sample_t(4, None)
    assert values.shape == (4,)
    assert torch.all((values >= 0.1) & (values <= 1.0))
    normalized = (values - 0.1) / 0.9
    assert torch.equal(torch.floor(normalized * 4), torch.arange(4).float())


def test_sample_t_selects_accumulation_partition(monkeypatch):
    model = _bare(
        Diffusion,
        antithetic_sampling=False,
        sampling_eps=0.0,
        config=SimpleNamespace(loader=SimpleNamespace(global_batch_size=8)),
    )
    model._test_trainer = SimpleNamespace(
        num_nodes=1, node_rank=0, num_devices=1, local_rank=0,
        accumulate_grad_batches=2,
    )
    monkeypatch.setattr(Diffusion, "trainer", property(lambda self: self._test_trainer))
    torch.manual_seed(12)
    full = torch.rand(8)
    torch.manual_seed(12)
    selected = model._sample_t(4, 1)
    assert torch.equal(selected, full.chunk(2)[1])


def test_sigma_from_alpha_and_discrete_time_nll_flow():
    model = _bare(Diffusion, T=4, loss_type="elbo")
    observed = {}
    model._sample_t = lambda *_args: torch.tensor([0.31, 0.74])
    model.noise = lambda t: (-torch.ones_like(t), 1 - 0.5 * t)
    def q_xt(x, alpha):
        observed["alpha"] = alpha
        return x

    model.q_xt = q_xt

    def forward(_self, xt, sigma):
        observed["sigma"] = sigma
        return torch.zeros(*xt.shape, 3)

    model.forward = MethodType(forward, model)
    model.nll_per_token = lambda **kwargs: kwargs["alpha_t"].expand(-1, 3)
    tokens = torch.zeros(2, 3, dtype=torch.long)
    result = model.nll(tokens, None)
    expected_t = torch.tensor([0.5, 0.75])
    expected_alpha = (1 - 0.5 * expected_t).unsqueeze(-1)
    assert torch.allclose(result, expected_alpha.expand(-1, 3))
    assert torch.allclose(observed["sigma"], -torch.log(expected_alpha))


def test_uniform_q_xt_endpoints_and_bos_preservation():
    model = _bare(UniformState, vocab_size=5, ignore_bos=True)
    tokens = torch.tensor([[1, 2, 3], [1, 4, 2]])
    clean = model.q_xt(tokens, torch.ones(2, 1))
    noisy = model.q_xt(tokens, torch.zeros(2, 1))
    assert torch.equal(clean, tokens)
    assert torch.equal(noisy[:, 0], tokens[:, 0])
    assert noisy.min() >= 0 and noisy.max() < 5


def test_uniform_prior_shape_dtype_and_range():
    model = _bare(UniformState, vocab_size=7)
    samples = model.prior_sample(3, 4)
    assert samples.shape == (3, 4)
    assert samples.dtype == torch.int64
    assert samples.min() >= 0 and samples.max() < 7


def test_absorbing_q_xt_and_prior_are_masked():
    model = _bare(AbsorbingState, mask_index=6, ignore_bos=True)
    tokens = torch.tensor([[1, 2, 3]])
    assert torch.equal(model.q_xt(tokens, torch.ones(1, 1)), tokens)
    noisy = model.q_xt(tokens, torch.zeros(1, 1))
    assert torch.equal(noisy, torch.tensor([[1, 6, 6]]))
    assert torch.equal(model.prior_sample(2, 3), torch.full((2, 3), 6))


def test_absorbing_transition_probabilities():
    model = _bare(AbsorbingState, mask_index=3, vocab_size=4)
    tokens = torch.tensor([[1, 3]])
    transition = model._transp_transition(tokens, torch.tensor([[0.7]]))
    assert transition.shape == (1, 2, 4)
    keep = torch.exp(torch.tensor(-0.7))
    move = 1 - keep
    assert torch.allclose(transition[0, 0], torch.tensor([0.0, keep, 0.0, 0.0]))
    assert torch.allclose(transition[0, 1], torch.tensor([move, move, move, 1.0]))


def test_staggered_score_does_not_mutate_input():
    model = _bare(AbsorbingState, mask_index=2)
    score = torch.tensor([[[1.0, 2.0, 3.0]]])
    original = score.clone()
    result = model._staggered_score(score, torch.tensor([[0.2]]))
    assert torch.equal(score, original)
    assert result.shape == score.shape
    assert torch.isfinite(result).all()


def test_semi_ar_restore_propagates_exception_and_restores_mode():
    model = _bare(Diffusion)
    model.config = SimpleNamespace(loader=SimpleNamespace(eval_batch_size=2))
    events = []
    model._eval_mode = lambda: events.append("eval")
    model._train_mode = lambda: events.append("train")
    model._semi_ar_sampler = lambda **_kwargs: (_ for _ in ()).throw(
        RuntimeError("semi-ar failed")
    )
    with pytest.raises(RuntimeError, match="semi-ar failed"):
        model.restore_model_and_semi_ar_sample(2, 3)
    assert events == ["eval", "train"]

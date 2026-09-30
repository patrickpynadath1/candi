from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from algos import algo_registry
from algos.replaid_hybrid import HybridClock
from dataloader import Text8Tokenizer
from samplers.base import SamplingRequest
from samplers.replaid_hybrid_ddpm import RePlaidHybridDDPMSampler


def make_model():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[3]/"configs"), version_base=None):
        config = compose(config_name="config", overrides=[
            "algo=replaid_hybrid", "model=tiny", "data=ptb", "training.ema=0",
            "model.length=4", "model.hidden_size=8", "model.cond_dim=8",
            "model.n_blocks=1", "model.n_heads=2", "model.dropout=0",
            "sampling.predictor=replaid_hybrid_ddpm"])
    return algo_registry.get(config.algo.name)(config, Text8Tokenizer())


def test_clock_normalization_monotonicity_and_importance_integral():
    clock = HybridClock(-4, 3, bins=4).double()
    with torch.no_grad():
        clock.clock_logits.copy_(torch.tensor([-1., 0., 1., 2.]))
    t = torch.linspace(0, 1, 10001, dtype=torch.float64)
    u, derivative = clock.inverse_cdf(t)
    assert u[0] == 0 and u[-1] == 1
    assert (u.diff() >= 0).all() and (derivative > 0).all()
    assert torch.trapezoid((1+u)*derivative, t).item() == pytest.approx(1.5, abs=.002)
    assert torch.allclose(clock(t[[0, -1]])[0], torch.tensor([-4., 3.], dtype=torch.float64))


def test_clock_score_gradient_matches_exact_second_moment_gradient():
    clock = HybridClock(-4, 3, bins=3).double()
    p = clock.probabilities()
    second = torch.tensor([1., 4., 9.], dtype=torch.float64)
    density = 3*p
    exact = (second/(9*p)).sum()
    expected = torch.autograd.grad(exact, clock.clock_logits, retain_graph=True)[0]
    estimate = second.sqrt()/density.detach()
    surrogate = (p.detach()*clock.variance_surrogate(estimate, density)).sum()
    actual = torch.autograd.grad(surrogate, clock.clock_logits)[0]
    assert surrogate.item() == 0
    assert torch.allclose(actual, expected)


def test_conditional_estimator_matches_exact_mask_enumeration():
    # Two correlated-position-dependent toy losses; enumerate masks exactly.
    for u in [.00001, .2, .8, 1.]:
        original = 0.
        conditional = 0.
        for m0 in [0, 1]:
            for m1 in [0, 1]:
                probability = (u if m0 else 1-u)*(u if m1 else 1-u)
                original += probability*(m0*((2+m1)/u+3) + m1*((4+m0)/u+5))/2
        for other in [0, 1]:
            probability = u if other else 1-u
            conditional += probability*((2+other+u*3)+(4+other+u*5))/2
        assert conditional == pytest.approx(original)


@pytest.mark.parametrize("batch", [1, 3])
def test_loss_backprop_context_and_sampling(batch, monkeypatch):
    model = make_model()
    tokens = torch.tensor([[1, 2, 3, 4]]).expand(batch, -1)
    valid = torch.tensor([[0, 0, 1, 1]]).expand(batch, -1)
    observed = []
    original = model.denoise
    def capture(z, gamma, context):
        observed.append((z.detach(), context.detach()))
        return original(z, gamma, context)
    monkeypatch.setattr(model, "denoise", capture)
    # Avoid the analytically solved, near-zero-noise region when asserting
    # strictly nonzero head gradients.
    monkeypatch.setattr(model.gamma_schedule, "draw", lambda count: (
        torch.full((count,), .8, dtype=torch.float64),
        (64*model.gamma_schedule.probabilities()[51]).expand(count)))
    result = model._loss(tokens, valid, train_mode=True)
    result.loss.backward()
    assert torch.isfinite(result.loss)
    for parameter in [model.backbone.token_embedding, model.backbone.output_layer.linear.weight,
                      *model.gamma_schedule.parameters()]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    clean = model.sampling_embeddings(tokens)
    assert torch.equal(observed[0][0][:, :2], clean[:, :2])
    assert torch.equal(observed[0][1][:, :2], clean[:, :2])
    sampler = RePlaidHybridDDPMSampler()
    sampler.validate_model(model)
    result = sampler.sample(model, SamplingRequest(num_samples=batch, num_steps=3, eps=0,
                              prompt_tokens=tokens, prompt_mask=~valid.bool()))
    assert torch.equal(result.samples[:, :2], tokens[:, :2])
    assert result.metadata["nfe"] == 3
    assert result.samples.min() >= 0 and result.samples.max() < model.vocab_size


def test_reveal_probabilities_include_exact_endpoints():
    sampler = RePlaidHybridDDPMSampler()
    assert torch.equal(sampler.reveal_probability(torch.tensor([1., .4]), torch.tensor([0., 0.])), torch.ones(2))
    assert sampler.reveal_probability(torch.tensor([.4]), torch.tensor([.2])).item() == pytest.approx(.5)


def test_all_context_returns_zero_loss():
    model = make_model()
    tokens = torch.ones(2, 4, dtype=torch.long)
    loss = model.nll(tokens, None, train_mode=True, valid_tokens=torch.zeros_like(tokens))
    assert torch.equal(loss, torch.zeros_like(loss))


def test_matched_sudoku_config_composes():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[3]/"configs"), version_base=None):
        config = compose(config_name="config", overrides=["+experiment=sudoku_replaid_hybrid_matched"])
    assert config.algo.name == "replaid_hybrid"
    assert config.wandb.project == "replaid-sudoku"
    assert config.sampling.predictor == "replaid_hybrid_ddpm"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_bf16_optimizer_updates_model_clock_and_endpoints():
    model = make_model().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tokens = torch.randint(1, 20, (16, 4), device="cuda")
    valid = torch.ones_like(tokens)
    before_head = model.backbone.output_layer.linear.weight.detach().clone()
    before_clock = model.gamma_schedule.clock_logits.detach().clone()
    before_endpoint = model.gamma_schedule.gamma_min.detach().clone()
    for _ in range(3):
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model._loss(tokens, valid, train_mode=True)
        assert torch.isfinite(result.loss)
        result.loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all()
        optimizer.step()
        model.backbone.project_embeddings_()
    assert not torch.equal(before_head, model.backbone.output_layer.linear.weight)
    assert not torch.equal(before_clock, model.gamma_schedule.clock_logits)
    assert not torch.equal(before_endpoint, model.gamma_schedule.gamma_min)
    replacement = make_model().cuda()
    replacement.load_state_dict(model.state_dict())
    t = torch.linspace(0, 1, 17, device="cuda")
    assert torch.equal(model.sampling_gamma(t), replacement.sampling_gamma(t))


def test_runtime_registry_in_fresh_interpreter():
    import subprocess
    import sys
    subprocess.run([sys.executable, '-c', '''
from types import SimpleNamespace
from samplers import create_sampler
config=SimpleNamespace(sampling=SimpleNamespace(predictor="replaid_hybrid_ddpm"))
assert create_sampler(config).spec.name == "replaid_hybrid_ddpm"
'''], check=True)


def test_public_generation_uses_zero_endpoint():
    model = make_model()
    samples = model.generate_samples(2, num_steps=2)
    assert samples.shape == (2, 4)


def test_rao_blackwell_mask_estimator_is_unbiased_and_lower_variance():
    import itertools
    n = 3
    for u in [.001, .2, .8, 1.]:
        exact = old_mean = new_mean = old_second = new_second = 0.
        for mask in itertools.product([0, 1], repeat=n):
            k = sum(mask)
            p = u**k*(1-u)**(n-k)
            f = [float((i+1)*(1+k)) for i in range(n)]
            exact += p*sum(m*f_i/u for m, f_i in zip(mask, f))/n
            if not k:
                continue
            proposal = p*k/(n*u)
            new = sum(m*f_i for m, f_i in zip(mask, f))/k
            new_mean += proposal*new
            new_second += proposal*new*new
            for i in range(n):
                if mask[i]:
                    old_mean += proposal*f[i]/k
                    old_second += proposal*f[i]*f[i]/k
        assert new_mean == pytest.approx(exact)
        assert old_mean == pytest.approx(exact)
        assert new_second <= old_second+1e-10


def test_stratified_times_cover_every_probability_stratum():
    clock = HybridClock(-4, 3, bins=4).double()
    clock.clock_logits.data.copy_(torch.tensor([-1., 0., 1., 2.]))
    u, density = clock.draw(64)
    p = clock.probabilities().detach()
    index = (u*4).long().clamp_max(3)
    left = torch.cat((p.new_zeros(1), p.cumsum(0)[:-1]))[index]
    cdf = left+(u*4-index)*p[index]
    assert torch.equal((cdf*64).long().sort().values, torch.arange(64))
    assert not u.requires_grad
    assert density.requires_grad


@pytest.mark.parametrize("bound", [0.0, 0.1])
def test_hybrid_eb_public_sampling_preserves_context(bound):
    from omegaconf import OmegaConf
    model = make_model()
    OmegaConf.update(model.config, "sampling.entropy_bound", bound, force_add=True)
    tokens = torch.tensor([[1, 2, 0, 0]])
    mask = torch.tensor([[True, True, False, False]])
    result = model.generate_samples(1, num_steps=4, prompt_tokens=tokens,
        prompt_mask=mask, sampler_name="replaid_hybrid_entropy_bounded", return_result=True)
    assert torch.equal(result.samples[:, :2], tokens[:, :2])
    assert 1 <= result.metadata["nfe"] <= 2
    assert sum(count[0] for count in result.metadata["accepted_per_step"]) == 2

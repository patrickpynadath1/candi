"""Independent estimator, gradient, clock and sampler checks for the hybrid port.

The REPLAiD hybrid is the replaid_hybrid preset over HybridDiffusion: the shared
clock is the algorithm's importance time sampler, read by both views."""
import itertools
import torch
import pytest
from hydra.utils import instantiate
from core.sweeps.configuration import compose
from core.noise_schedules.gaussian_path import path_from_gamma
from core.objectives.hybrid_elbo import HybridELBO
from core.objectives.weighting import ObjectiveContext
from core.processes.absorbing import AbsorbingProcess
from core.time_sampling.importance import PiecewiseImportanceTimeSampler as PiecewiseInverseCDFClock
from core.training.module import TrainingModule


def config():
    return compose('replaid_hybrid', ['model.hidden_size=32','model.cond_dim=16',
                   'model.n_blocks=1','model.n_heads=4','task.vocab_size=5',
                   'task.length=4','model.dropout=0.0','training.ema=0.9'])


def test_clock_inverse_cdf_and_endpoints():
    clock = PiecewiseInverseCDFClock(bins=3, uniform_mix=.1)
    with torch.no_grad():
        clock.clock_logits.copy_(torch.tensor([-2.,1.,3.]))
    p = clock.probabilities()
    t = torch.cat((torch.zeros(1), p.cumsum(0)))
    u, _ = clock.inverse_cdf(t.clamp(0,1))
    torch.testing.assert_close(u, torch.linspace(0,1,4).double())
    assert clock.inverse_cdf(torch.tensor([0.,1.]))[0].tolist() == [0.,1.]
    torch.manual_seed(4)
    values, density = clock.draw(9000)
    counts = torch.bincount((values*3).long(), minlength=3)/9000
    torch.testing.assert_close(counts.double(), p, atol=.002, rtol=0)
    assert torch.all(density > 0)


def test_variance_surrogate_has_second_moment_gradient_not_loss_value():
    clock = PiecewiseInverseCDFClock(bins=2)
    p = clock.probabilities()
    integrand = torch.tensor([1.,3.], dtype=torch.float64)
    estimate = integrand/(2*p)
    surrogate = (p.detach()*clock.variance_surrogate(estimate,2*p)).sum()
    assert surrogate.item() == 0
    expected = (integrand.square()/p).sum()/4
    actual_grad = torch.autograd.grad(surrogate, clock.clock_logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, clock.clock_logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_forced_target_estimator_matches_enumerated_original_mask_law():
    objective = HybridELBO()
    u = .3
    tokens = torch.tensor([[0,1]])
    log_probs = torch.tensor([[[.8,.2],[.6,.4]]]).log()
    clean = torch.nn.functional.one_hot(tokens,2).float()
    predicted = log_probs.exp()
    gamma = torch.tensor([0.],dtype=torch.float64)
    lower, upper = torch.tensor(-1.), torch.tensor(1.)
    ce = -log_probs.gather(-1,tokens[...,None]).squeeze(-1).double()
    mse = (predicted-clean).double().square().sum(-1)

    class OneHotCodebook:
        def encode(self, tokens):
            return torch.nn.functional.one_hot(tokens, 2).float()

        def posterior_mean(self, probabilities):
            return probabilities.float()

    # Gaussian log-SNR gamma(u) = lower + (upper - lower) u at u (gamma = 0),
    # absorbing keep rate 1 - u; the terminal prior reads gamma(1) = upper.
    def objective_at(mask):
        state = torch.where(mask, 2, tokens)
        context = ObjectiveContext(
            time=torch.tensor([u]),
            processes=(None, AbsorbingProcess(2)),
            coefficients=(path_from_gamma(gamma, (upper-lower).double()[None]),
                          (torch.tensor([1-u], dtype=torch.float64), -torch.ones(1, dtype=torch.float64))),
            states=(None, state),
            representations=(OneHotCodebook(), None),
            schedules=(lambda t: path_from_gamma(upper.double().expand(len(t)), torch.ones_like(t)), None),
            loss_mask=torch.ones_like(mask), corruption_mask=torch.ones_like(mask),
            time_density=torch.ones(1, dtype=torch.float64),
        )
        return objective(log_probs, tokens, context=context).total

    # Prior is independent of the mask; obtain it with the no-mask case.
    prior = objective_at(torch.zeros_like(tokens).bool())
    original = 0.
    forced = 0.
    for bits in itertools.product([False,True], repeat=2):
        mask = torch.tensor([bits])
        k = sum(bits)
        probability = u**k*(1-u)**(2-k)
        original += probability*((ce/u + mse)*mask).mean()
        if k:
            proposal_probability = probability*k/(2*u)
            loss = objective_at(mask)
            forced += proposal_probability*(loss-prior).mean()
    torch.testing.assert_close(forced,original,rtol=1e-6,atol=1e-7)


def test_train_checkpoint_generate_and_nfe(tmp_path):
    cfg = config()
    module = TrainingModule(cfg)
    algo = module.algorithm
    x = torch.tensor([[1,2,3,4],[4,3,2,1]])
    valid = torch.tensor([[False,True,True,True],[False,True,True,True]])
    loss = algo.loss(x,valid,corruption_mask=valid,training=True)['loss']
    loss.backward()
    for name,p in algo.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert algo.time_sampler.clock_logits.grad.abs().sum() > 0
    checkpoint = tmp_path/'state.pt'
    torch.save(algo.state_dict(),checkpoint)
    clone = TrainingModule(cfg).algorithm
    clone.load_state_dict(torch.load(checkpoint,weights_only=True),strict=True)
    calls=[]
    hook = clone.model.register_forward_hook(lambda *args: calls.append(1))
    sampler=instantiate(cfg.sampler,steps=3)
    generated,info=clone.generate(sampler,x,~valid)
    hook.remove()
    assert len(calls)==info['nfe']==3
    assert info['nfe_per_sample']==[3,3]
    assert sampler.steps_for_nfe(3)==3
    assert torch.equal(generated[~valid],x[~valid])
    assert generated.min() >= 0 and generated.max() < 5
    # Both views must read the same learned clock, including after reload.
    assert clone.adaptive_time_views == (0, 1)
    assert torch.equal(clone.time_sampler.clock_logits, algo.time_sampler.clock_logits)
    t = torch.linspace(0, 1, 7, dtype=torch.float64)
    gaussian, reveal = clone.points(t)
    assert torch.equal(gaussian.time, reveal.time)
    assert torch.equal(gaussian.time, clone.time_sampler(t))


def test_validation_does_not_train_clock_logits():
    a=instantiate(config().algorithm)
    x=torch.randint(5,(2,4))
    a.loss(x,torch.ones_like(x).bool(),training=False)['loss'].backward()
    assert a.time_sampler.clock_logits.grad is None


def test_mask_contract():
    a=instantiate(config().algorithm)
    x=torch.ones(2,4,dtype=torch.long)
    with pytest.raises(ValueError,match='binary'):
        a.loss(x,torch.full((2,4),.5))
    with pytest.raises(ValueError,match='corruption_mask'):
        a.loss(x,torch.ones_like(x),corruption_mask=torch.zeros_like(x))


def test_lightning_two_updates_then_generate(tmp_path):
    import lightning as L
    from torch.utils.data import DataLoader
    cfg=config()
    cfg.output_dir=str(tmp_path)
    cfg.training.warmup_steps=0
    module=TrainingModule(cfg)
    samples=[{'input_ids':torch.tensor([1,2,3,4]),
              'loss_mask':torch.tensor([False,True,True,True]),
              'corruption_mask':torch.tensor([False,True,True,True])} for _ in range(4)]
    loader=DataLoader(samples,batch_size=2)
    trainer=L.Trainer(accelerator='cpu',devices=1,max_steps=2,logger=False,
                      enable_checkpointing=False,enable_progress_bar=False,
                      enable_model_summary=False,num_sanity_val_steps=0)
    trainer.fit(module,loader)
    assert trainer.global_step==2
    assert torch.isfinite(module.algorithm.time_sampler.clock_logits).all()
    x=torch.tensor([[1,2,3,4]])
    fixed=torch.tensor([[True,False,False,False]])
    generated,_=module.algorithm.generate(instantiate(cfg.sampler,steps=1),x,fixed)
    assert generated[0,0]==1


def test_high_noise_time_conditioning_is_not_clipped():
    from core.models.modular import TIME_COORDINATES
    from core.modifiers.hybrid.views import ViewPoint
    from core.noise_schedules.gaussian_path import path_from_gamma
    cfg = config()
    gamma = torch.tensor([-13.3, 46.6, 64.6, 67.3], dtype=torch.float64)
    point = ViewPoint(torch.ones_like(gamma), path_from_gamma(gamma, torch.ones_like(gamma)))
    actual = TIME_COORDINATES[cfg.model.time.coordinate](point.time, point, cfg.model.time.epsilon)
    torch.testing.assert_close(actual, gamma.float())


def test_block_layout_is_rejected():
    a = instantiate(config().algorithm)
    x = torch.ones(2, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="block sequence layouts"):
        a.loss(x, torch.ones_like(x), sequence_layout=object())

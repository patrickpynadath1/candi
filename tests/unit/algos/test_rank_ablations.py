from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import main  # noqa: F401 -- registers Hydra resolvers
from algos import algo_registry
from algos.noise_schedules import CDCDAdaptiveSchedule
from samplers import create_sampler
from tasks.sudoku import SudokuTokenizer


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = OmegaConf.load(ROOT / "configs/sweeps/sudoku_rank.yaml")


def config_for(variant, mode):
    entry = MANIFEST.variants[variant]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(config_name="config", overrides=[
            f"+experiment={entry.experiment}", *entry.overrides,
            f"algo.pure_continuous={'true' if mode == 'pure' else 'false'}",
            "training.ema=0", "model.length=8", "model.hidden_size=16",
            "model.cond_dim=16", "model.n_blocks=1", "model.n_heads=2",
            "model.dropout=0", "wandb=null",
        ])


@pytest.mark.parametrize("variant,mode", [(v,m) for v,e in MANIFEST.variants.items() for m in e.modes])
def test_real_rank_ablation_backward_and_sampling(variant, mode):
    config = config_for(variant, mode)
    model = algo_registry.get(config.algo.name)(config, SudokuTokenizer())
    create_sampler(config).validate_model(model)
    tokens = torch.randint(1, 10, (2, 8))
    valid = torch.ones_like(tokens)
    valid[:, :3] = 0
    loss = model._loss(tokens, valid).loss
    assert torch.isfinite(loss)
    loss.backward()
    gradient = model.backbone.output_embedding.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    model.eval()
    result = model.generate_samples(
        2, num_steps=4, prompt_tokens=tokens, prompt_mask=~valid.bool(),
        return_result=True,
    )
    assert result.metadata["nfe"] == 4
    assert result.samples.shape == tokens.shape
    assert torch.equal(result.samples[:, :3], tokens[:, :3])
    assert result.samples.min() >= 0 and result.samples.max() < 12


@pytest.mark.parametrize("mode", ["pure", "hybrid"])
def test_candi_mask_endpoints_and_rank_endpoints(mode):
    config = config_for("candi_geometric", mode)
    model = algo_registry.get(config.algo.name)(config, SudokuTokenizer())
    tokens = torch.ones(100_000, 1, dtype=torch.long)
    at_noise = model.discrete_noising(tokens, torch.zeros(100_000, 1))
    assert (at_noise == model.mask_index).all()
    at_clean = model.discrete_noising(tokens, torch.ones(100_000, 1))
    assert (at_clean == (model.mask_index if mode == "pure" else 1)).all()
    sigma = model.sampling_continuous_schedule(torch.tensor([0., 1.]), 1)
    assert sigma[-1].item() == pytest.approx(28.2065, rel=1e-4)


def test_embedding_sampler_clamps_embedding_prompt_every_step():
    config = config_for("embedding_linear", "hybrid")
    model = algo_registry.get(config.algo.name)(config, SudokuTokenizer()).eval()
    tokens = torch.tensor([[11, 1, 2, 3, 4, 5, 6, 7]])
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[:, :3] = True
    original = model.sample_continuous_transition
    observed = []

    def check(state, *args, **kwargs):
        assert torch.equal(state[:, :3], model.backbone.get_embedding(tokens)[:, :3])
        observed.append(True)
        return original(state, *args, **kwargs)

    model.sample_continuous_transition = check
    model.generate_samples(1, num_steps=4, prompt_tokens=tokens, prompt_mask=mask)
    assert len(observed) == 4


def test_adaptive_candi_records_raw_ce():
    config = config_for("candi_geometric_adaptive", "hybrid")
    model = algo_registry.get(config.algo.name)(config, SudokuTokenizer())
    captured = []
    model._record_time_loss = lambda t, loss, *args, **kwargs: captured.append(loss)
    model._sample_t = lambda *args: torch.tensor([.2, .8])
    tokens = torch.randint(1, 10, (2, 8))
    unweighted = model.nll(tokens, None, train_mode=True)
    assert model.hybrid_loss_weighting == "none"
    assert torch.allclose(unweighted, captured[0])


def test_hybrid_adaptive_schedule_averages_only_unrevealed_valid_tokens():
    config = config_for("candi_geometric_adaptive", "hybrid")
    model = algo_registry.get(config.algo.name)(config, SudokuTokenizer())
    captured = []
    model._record_time_loss = lambda _t, _loss, valid, *_args, **_kwargs: (
        captured.append(valid.clone())
    )
    reveal = torch.tensor([
        [True, False, False, True, False, False, True, False],
        [False, True, False, False, True, False, False, True],
    ])
    valid = torch.tensor([
        [1, 1, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 1, 1, 1, 0, 1],
    ])
    model._sample_t = lambda *args: torch.tensor([.2, .8])
    original_q_xt = model.q_xt

    def fixed_reveal(*args, **kwargs):
        noisy = original_q_xt(*args, **kwargs)
        noisy["reveal_mask"] = reveal
        return noisy

    model.q_xt = fixed_reveal
    tokens = torch.randint(1, 10, (2, 8))
    model.nll(tokens, None, train_mode=True, valid_tokens=valid)
    assert len(captured) == 1
    assert torch.equal(captured[0], valid.bool() & ~reveal)


def test_adaptive_recorder_omits_examples_without_observed_tokens():
    config = config_for("candi_geometric_adaptive", "hybrid")
    model = algo_registry.get(config.algo.name)(config, SudokuTokenizer())

    class Recorder:
        def __init__(self):
            self.calls = []

        def record_time_loss_pair(self, times, losses, step):
            self.calls.append((times, losses, step))

    recorder = Recorder()
    times = torch.tensor([.1, .2, .3])
    losses = torch.tensor([[9., 9.], [2., 4.], [6., 12.]])
    observed = torch.tensor([[0, 0], [1, 1], [0, 1]])
    model._record_time_loss(
        times, losses, observed, True, schedule=recorder
    )
    assert len(recorder.calls) == 1
    recorded_times, recorded_losses, _ = recorder.calls[0]
    assert torch.equal(recorded_times, torch.tensor([.2, .3]))
    assert torch.equal(recorded_losses, torch.tensor([3., 12.]))


@pytest.mark.parametrize(
    "variant",
    ["candi_linear_cdcd_adaptive", "candi_geometric_cdcd_adaptive"],
)
def test_cdcd_variants_instantiate_cdcd_schedule(variant):
    config = config_for(variant, "hybrid")
    model = algo_registry.get(config.algo.name)(config, SudokuTokenizer())
    assert config.noise.adaptive is True
    assert isinstance(model.noise, CDCDAdaptiveSchedule)


@pytest.mark.parametrize(
    "variant",
    [
        "candi_geometric_adaptive",
        "embedding_geometric_rank_adaptive",
        "slerp_geometric_rank_adaptive",
    ],
)
def test_hybrid_likelihood_weighting_is_an_optional_ablation(variant):
    config = config_for(variant, "hybrid")
    config.algo.hybrid_loss_weighting = "likelihood"
    model = algo_registry.get(config.algo.name)(config, SudokuTokenizer())
    assert model.hybrid_loss_weighting == "likelihood"


@pytest.mark.parametrize("variant", ["candi_geometric_adaptive", "embedding_linear_adaptive"])
def test_coupled_algorithms_reject_factorized_schedule(variant):
    config = config_for(variant, "hybrid")
    OmegaConf.set_struct(config.algo, False)
    config.algo.factorized_adaptive_schedule = True
    with pytest.raises(ValueError, match="single linear corruption path|one coupled time axis"):
        algo_registry.get(config.algo.name)(config, SudokuTokenizer())


@pytest.mark.parametrize('mode', ['pure', 'hybrid'])
def test_slerp_states_remain_spherical_and_coupling_survives_refit(mode, tmp_path):
    config = config_for('slerp_geometric_rank_adaptive', mode)
    config.checkpointing.save_dir = str(tmp_path)
    config.noise.adaptive_buffer_size = 8
    config.noise.adaptive_refit_every = 1
    config.noise.adaptive_warmup_steps = 0
    config.noise.adaptive_n_knots = 3
    model = algo_registry.get(config.algo.name)(config, SudokuTokenizer())
    model._sample_t = lambda n, *_args: torch.full((n,), .8)
    tokens = torch.randint(1, 10, (8, 8))
    alpha = torch.linspace(.05, .95, 8)[:, None]
    noisy = model.q_xt(tokens, alpha)
    assert torch.allclose(noisy['xt'].norm(dim=-1), torch.ones(8, 8), atol=1e-6)
    rho = 1-alpha[:, 0]
    assert torch.allclose(model.rank_schedule.expected_rank(noisy['continuous_noise']), .01*49**rho, atol=1e-6)
    model._loss(tokens, torch.ones_like(tokens), train_mode=True).loss.backward()
    assert model.noise.refit_count == 1
    t = torch.linspace(0, 1, 21)
    rho = 1-model.noise.alpha_t(t)
    assert torch.allclose(model.rank_schedule.expected_rank(model.interpolation(rho)), .01*49**rho, atol=1e-6)
    original = model.sample_continuous_transition
    def check(state, *args, **kwargs):
        assert torch.allclose(state.norm(dim=-1), torch.ones_like(state[..., 0]), atol=1e-6)
        out, probs = original(state, *args, **kwargs)
        assert torch.allclose(out.norm(dim=-1), torch.ones_like(out[..., 0]), atol=1e-6)
        return out, probs
    model.sample_continuous_transition = check
    model.eval().generate_samples(2, num_steps=4)


def test_requested_matrix_has_eight_pure_and_eleven_hybrid_configs():
    from types import SimpleNamespace
    from new_scripts.sudoku_rank_ablation import entries
    args = SimpleNamespace(variants=None, modes=None, difficulties=None, seeds=None)
    jobs = entries(MANIFEST, args)
    assert len(jobs) == 171
    pure = {j['variant'] for j in jobs if j['mode'] == 'pure'}
    hybrid = {j['variant'] for j in jobs if j['mode'] == 'hybrid'}
    assert pure == {'embedding_linear_adaptive', 'embedding_geometric_rank',
                    'embedding_linear_rank', 'slerp_truncated_adaptive',
                    'slerp_geometric_rank', 'onehot_gaussian_adaptive',
                    'candi_geometric', 'candi_orig'}
    assert hybrid == {'embedding_linear', 'embedding_linear_rank',
                      'embedding_geometric_rank', 'embedding_geometric_rank_adaptive',
                      'slerp_geometric_rank_adaptive', 'candi_orig',
                      'candi_geometric', 'candi_geometric_adaptive',
                      'candi_linear_adaptive',
                      'candi_geometric_cdcd_adaptive',
                      'candi_linear_cdcd_adaptive'}
    args.modes = ['hybrid']
    assert len(entries(MANIFEST, args)) == 99


@pytest.mark.parametrize('variant', ['onehot_gaussian_adaptive', 'embedding_linear_adaptive'])
def test_gaussian_adaptive_uses_unranked_interpolation(variant, tmp_path):
    config = config_for(variant, 'pure')
    config.checkpointing.save_dir = str(tmp_path)
    config.noise.adaptive_buffer_size = 8
    config.noise.adaptive_refit_every = 1
    config.noise.adaptive_warmup_steps = 0
    config.noise.adaptive_n_knots = 3
    model = algo_registry.get(config.algo.name)(config, SudokuTokenizer())
    assert not hasattr(model, 'rank_schedule')
    tokens = torch.randint(1, 10, (8, 8))
    rho = torch.linspace(0, 1, 8)
    noisy = model.q_xt(tokens, (1-rho)[:,None])
    assert torch.allclose(noisy['continuous_noise'], rho)
    assert torch.equal(noisy['xt'][0], model.clean_state(tokens)[0])
    observations = []
    original = model.noise.record_time_loss_pair
    def record(t, loss, step):
        observations.append((t.clone(), loss.clone()))
        original(t, loss, step)
    model.noise.record_time_loss_pair = record
    model._loss(tokens, torch.ones_like(tokens), train_mode=True)
    assert observations and model.noise.refit_count == 1

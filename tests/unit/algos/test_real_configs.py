from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

import main  # noqa: F401 -- registers the config resolvers used by config.yaml
from algos import algo_registry
from dataloader import Text8Tokenizer
from samplers import create_sampler


CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")


class OfflineGPT2Tokenizer:
    vocab_size = 50_257
    bos_token_id = eos_token_id = 50_256
    mask_token = None
    mask_token_id = None
    added_tokens_decoder = {}

    def __len__(self):
        return self.vocab_size


def _tokenizer_for(algo):
    # DUO's checked-in integral table is specifically calibrated for GPT-2's
    # 50,257-token vocabulary. Using the tiny text8 tokenizer here would make
    # alpha(t) negative and hide a real configuration invariant behind NaNs.
    if algo in {"duo", "distillation"}:
        return OfflineGPT2Tokenizer()
    return Text8Tokenizer()

ALGORITHM_CONFIGS = [
    ("ar", ("sampling.predictor=ar",)),
    ("block_diffusion", ("sampling.predictor=block_diffusion",)),
    ("mdlm", ("sampling.predictor=ancestral_cache",)),
    ("d3pm", ("sampling.predictor=ancestral",)),
    ("sedd", ("sampling.predictor=analytic",)),
    ("cadd", ("sampling.predictor=cadd",)),
    ("candi", ("sampling.predictor=candi_cached",)),
    ("candi_looped", ("sampling.predictor=candi_looped",)),
    ("candi_register_looped", ("sampling.predictor=candi_looped",)),
    ("candi_softmax_looped", ("sampling.predictor=candi_looped_entropy_bounded",)),
    ("candi_rank", ("sampling.predictor=candi", "noise=rank-adaptive")),
    ("candi_embedding", ("sampling.predictor=candi_embedding", "noise=rank-adaptive")),
    ("candi_blurred", ("sampling.predictor=candi_entropy_bounded",)),
    ("candi_ra_change", ("sampling.predictor=candi",)),
    ("cont_diff_onehot", ("sampling.predictor=continuous_onehot",)),
    ("embedding_diffusion", ("sampling.predictor=continuous_embedding",)),
    ("blurring_diffusion", ("sampling.predictor=blurring_ode",)),
    ("cfm", ("sampling.predictor=semicat_euler",)),
    ("semicat", ("sampling.predictor=semicat_euler",)),
    ("mdlm_perm", ("sampling.predictor=permuted_ar",)),
    ("duo_base", ("sampling.predictor=ancestral",)),
    (
        "duo",
        (
            "sampling.predictor=ancestral",
            f"algo.integral_cache_path={CONFIG_DIR}/../integral/gpt2.pkl",
        ),
    ),
    ("flm", ("sampling.predictor=flm_euler",)),
    ("sfm", ("sampling.predictor=sfm",)),
    ("replaid", ("sampling.predictor=replaid_ddpm",)),
    ("replaid_keep_mse", ("sampling.predictor=replaid_keep_ddpm",)),
    ("replaid_keep_hybrid_elbo", ("sampling.predictor=replaid_keep_ddpm",)),
    ("replaid_keep_mse_reveal_ce", ("sampling.predictor=replaid_keep_ddpm",)),
    (
        "latent_fm",
        (
            "model=latent_split_tiny",
            "model.n_blocks=2",
            "sampling.predictor=latent_fm",
        ),
    ),
    (
        "distillation",
        (
            "sampling.predictor=ancestral",
            f"algo.integral_cache_path={CONFIG_DIR}/../integral/gpt2.pkl",
        ),
    ),
]

TIME_CONDITIONED_CONFIGS = [
    entry for entry in ALGORITHM_CONFIGS
    if entry[0] not in {"ar", "block_diffusion", "mdlm_perm"}
]


def _compose_tiny(algo, *extra_overrides):
    overrides = [
        f"algo={algo}",
        "model=tiny",
        "data=ptb",
        "training.ema=0",
        "model.length=4",
        "model.hidden_size=8",
        "model.cond_dim=8",
        "model.n_blocks=1",
        "model.n_heads=2",
        "model.dropout=0",
        *extra_overrides,
    ]
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="config", overrides=overrides)


@pytest.mark.parametrize(
    ("algo", "overrides"),
    ALGORITHM_CONFIGS,
)
def test_shipping_algorithm_config_constructs_real_backbone_and_sampler(algo, overrides):
    config = _compose_tiny(algo, *overrides)
    model = algo_registry.get(config.algo.name)(config, _tokenizer_for(algo))
    sampler = create_sampler(config)

    expected_blocks = 2 if algo == "latent_fm" else 1
    assert len(model.backbone.blocks) == expected_blocks
    assert model.backbone.blocks[0].attn_qkv.weight.shape[1] == 8
    sampler.validate_model(model)


def test_real_mdlm_training_path_reaches_output_projection():
    torch.manual_seed(8)
    config = _compose_tiny("mdlm", "sampling.predictor=ancestral_cache")
    model = algo_registry.get("mdlm")(config, Text8Tokenizer())
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    valid = torch.ones_like(tokens)

    loss = model._loss(tokens, valid).loss
    loss.backward()

    gradient = model.backbone.output_layer.linear.weight.grad
    assert torch.isfinite(loss)
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_block_diffusion_runs_rolling_sampler_with_both_updates():
    config = _compose_tiny(
        "block_diffusion",
        "sampling.predictor=block_diffusion",
        "algo.block_size=2",
    )
    model = algo_registry.get("block_diffusion")(config, Text8Tokenizer()).eval()
    for update in ("euler", "renoise"):
        model.config.algo.rolling_update = update
        result = model.generate_samples(
            2,
            return_result=True,
            sampler_overrides={
                "predictor": "block_diffusion",
                "use_float64": False,
            },
            sampler_name="block_diffusion",
        )
        assert result.samples.shape == (2, 4)
        assert result.metadata["nfe"] > 0
        assert result.metadata["verification_sequences"] > 0
        assert 0 <= result.metadata["mean_accepted_length"] <= 2
        assert 1 <= result.metadata["mean_verified_advance"] <= 2


def test_block_diffusion_verifier_free_euler_honors_fixed_and_budgeted_nfe():
    config = _compose_tiny(
        "block_diffusion",
        "sampling.predictor=block_diffusion_euler",
        "algo.block_size=2",
        "algo.rolling_warmup_steps=0",
        "model.length=5",
    )
    model = algo_registry.get("block_diffusion")(config, Text8Tokenizer()).eval()
    for overrides in (
        {"tokens_per_step": 2, "nfe_budget": None, "refinement_steps": 0},
        {"tokens_per_step": 1, "nfe_budget": 2, "refinement_steps": 0},
        {"tokens_per_step": 2, "nfe_budget": None, "refinement_steps": 1},
    ):
        model.config.algo.euler_tokens_per_step = overrides["tokens_per_step"]
        model.config.algo.euler_nfe_budget = overrides["nfe_budget"]
        model.config.algo.rolling_refinement_steps = overrides["refinement_steps"]
        result = model.generate_samples(
            2,
            return_result=True,
            sampler_name="block_diffusion_euler",
            sampler_overrides={
                "predictor": "block_diffusion_euler",
                "use_float64": False,
            },
        )
        assert result.samples.shape == (2, 5)
        expected_nfe = 3 if overrides["refinement_steps"] else 2
        assert result.metadata["nfe"] == expected_nfe
        assert result.metadata["speculative"] is False


def test_mdlm_can_allocate_mask_after_added_pad_token():
    class TokenizerWithAddedPad(Text8Tokenizer):
        vocab_size = 35
        pad_token_id = 35
        mask_token = None
        mask_token_id = None

        def __len__(self):
            return 36

    config = _compose_tiny(
        "mdlm",
        "sampling.predictor=ancestral_cache",
        "algo.mask_after_added_tokens=true",
    )
    model = algo_registry.get("mdlm")(config, TokenizerWithAddedPad())

    assert model.mask_index == 36
    assert model.mask_index != model.tokenizer.pad_token_id
    assert model.vocab_size == 37


def test_duo_rejects_integral_cache_for_a_different_vocabulary():
    config = _compose_tiny(
        "duo",
        "sampling.predictor=ancestral",
        f"algo.integral_cache_path={CONFIG_DIR}/../integral/gpt2.pkl",
    )
    with pytest.raises(ValueError, match="integral cache vocabulary mismatch"):
        algo_registry.get("duo")(config, Text8Tokenizer())


@pytest.mark.parametrize(
    ("algo", "overrides"),
    ALGORITHM_CONFIGS,
)
def test_shipping_algorithm_runs_real_loss_and_backward(algo, overrides):
    torch.manual_seed(9)
    config = _compose_tiny(algo, *overrides)
    model = algo_registry.get(config.algo.name)(config, _tokenizer_for(algo))
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    result = model._loss(tokens, torch.ones_like(tokens))
    result.loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.backbone.parameters()
        if parameter.grad is not None
    ]
    assert torch.isfinite(result.loss)
    assert 0 < result.num_tokens.item() <= tokens.numel()
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    if algo == "distillation":
        assert model.teacher is not None
        assert all(parameter.grad is None for parameter in model.teacher.parameters())
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


@pytest.mark.parametrize(("algo", "overrides"), TIME_CONDITIONED_CONFIGS)
def test_time_conditioned_algorithm_reports_adaptive_statistic(algo, overrides):
    """Every scheduled objective must report the time/statistic pair it used."""
    adaptive_overrides = [
        *overrides,
        "noise=rank-adaptive" if algo in {"candi_rank", "candi_embedding"} else "noise=log-linear-adaptive",
        "noise.adaptive_buffer_size=2",
        "noise.adaptive_refit_every=1",
        "noise.adaptive_warmup_steps=0",
        "noise.adaptive_n_grid=32",
        "noise.adaptive_n_knots=4",
        "noise.adaptive_spline_degree=2",
    ]
    if algo == "sfm":
        adaptive_overrides.append("algo.invert_time_convention=false")
    config = _compose_tiny(algo, *adaptive_overrides)
    model = algo_registry.get(config.algo.name)(config, _tokenizer_for(algo))
    assert hasattr(model.noise, "record_time_loss_pair")

    observations = []
    model._record_time_loss = lambda t, loss, *_args, **kwargs: observations.append(
        (t.detach(), loss.detach(), kwargs.get("schedule_name", "schedule"))
    )
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    result = model._loss(tokens, torch.ones_like(tokens), train_mode=True)

    assert torch.isfinite(result.loss)
    assert observations, f"{algo} bypassed the adaptive statistic recorder"
    for times, losses, _name in observations:
        assert times.shape == (tokens.shape[0],)
        assert losses.shape == tokens.shape
        assert torch.isfinite(times).all()
        assert torch.isfinite(losses).all()


def test_blurred_factorized_candi_reports_both_adaptive_axes():
    config = _compose_tiny(
        "candi_ra_change",
        "sampling.predictor=candi",
        "noise=log-linear-adaptive",
        "noise.adaptive_buffer_size=2",
        "noise.adaptive_refit_every=1",
        "noise.adaptive_warmup_steps=0",
        "noise.adaptive_n_grid=32",
        "noise.adaptive_n_knots=4",
        "noise.adaptive_spline_degree=2",
        "algo.factorized_adaptive_schedule=true",
        "algo.joint_loss_probe=false",
        "algo.spatial_blurring.bandwidth=0.5",
    )
    model = algo_registry.get("candi_ra_change")(config, Text8Tokenizer())
    observations = []
    model._record_time_loss = lambda _t, _loss, *_args, **kwargs: observations.append(
        kwargs.get("schedule_name", "schedule")
    )
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    result = model._loss(tokens, torch.ones_like(tokens), train_mode=True)

    assert torch.isfinite(result.loss)
    assert observations == ["schedule/discrete", "schedule/continuous_rank"]


@pytest.mark.parametrize(
    ("experiment", "adaptive", "factorized"),
    [
        ("sudoku_candi_ra_change", False, False),
        ("sudoku_candi_ra_adaptive", True, False),
        ("sudoku_candi_ra_factorized_adaptive", True, True),
    ],
)
def test_blurred_candi_experiment_composition(experiment, adaptive, factorized):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        config = compose(
            config_name="config", overrides=[f"+experiment={experiment}"]
        )
    assert config.algo.name == "candi_ra_change"
    assert config.algo.curve_type == "linear"
    assert config.algo.spatial_blurring.bandwidth == 0.5
    assert config.noise.get("adaptive", False) is adaptive
    assert config.algo.factorized_adaptive_schedule is factorized


@pytest.mark.parametrize('backbone', ['sphere_dit', 'sphere_arch'])
def test_sfm_corrected_adaptive_training_checkpoint_and_sampling(backbone, tmp_path):
    from samplers.base import SamplingRequest
    config = _compose_tiny('sfm', 'sampling.predictor=sfm',
        'model=small_sphere_arch' if backbone == 'sphere_arch' else 'model=tiny_sphere_dit',
        'model.hidden_size=8', 'model.cond_dim=8', 'model.n_blocks=1',
        'model.n_heads=2', 'model.length=4', 'model.dropout=0',
        f'algo.backbone={backbone}', f'algo.renormalize_weights={str(backbone == "sphere_arch").lower()}',
        f'checkpointing.save_dir={tmp_path}',
        'algo.invert_time_convention=false', 'noise=log-linear-adaptive',
        '+noise.adaptive_coordinate=physical', 'noise.alpha_max=0.093',
        'noise.adaptive_buffer_size=2', 'noise.adaptive_refit_every=1',
        'noise.adaptive_warmup_steps=0', 'noise.adaptive_n_grid=32',
        'noise.adaptive_n_knots=4', 'noise.adaptive_spline_degree=2')
    model = algo_registry.get('sfm')(config, Text8Tokenizer())
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    loss = model._loss(tokens, torch.ones_like(tokens), train_mode=True).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.backbone.sphere_embed.weight.grad).all()
    assert model.noise.refit_count == 1
    expected = model.noise.alpha_t(torch.linspace(0, 1, 10))
    restored = algo_registry.get('sfm')(config, Text8Tokenizer())
    restored.load_state_dict(model.state_dict())
    assert torch.equal(restored.noise.alpha_t(torch.linspace(0, 1, 10)), expected)
    with torch.no_grad():
        result = create_sampler(config).sample(restored.eval(), SamplingRequest(2, num_steps=3))
    assert result.samples.shape == tokens.shape
    assert result.metadata['nfe'] == 3


def test_sfm_sphere_dit_supports_smaller_spherical_input_dimension(tmp_path):
    from samplers.base import SamplingRequest
    config = _compose_tiny(
        'sfm', 'sampling.predictor=sfm', 'model=tiny_sphere_dit',
        'model.input_embedding_dim=16', 'model.hidden_size=32',
        'model.cond_dim=16', 'model.n_blocks=1', 'model.n_heads=4',
        'model.length=4', 'model.dropout=0', 'algo.backbone=sphere_dit',
        f'checkpointing.save_dir={tmp_path}', 'algo.invert_time_convention=false',
        'algo.auto_truncation=true',
    )
    model = algo_registry.get('sfm')(config, Text8Tokenizer())
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    loss = model._loss(tokens, torch.ones_like(tokens), train_mode=True).loss
    loss.backward()
    assert torch.isfinite(loss)
    from algos.spherical_utils import spherical_truncation_alpha_max
    assert config.noise.alpha_max == pytest.approx(
        spherical_truncation_alpha_max(model.vocab_size, 16, 0.1)
    )
    assert model.prior_sample(2, 4).shape == (2, 4, 16)
    with torch.no_grad():
        result = create_sampler(config).sample(
            model.eval(), SamplingRequest(2, num_steps=3)
        )
    assert result.samples.shape == tokens.shape


@pytest.mark.parametrize('experiment', ['sudoku_sfm_adaptive', 'sudoku_sfm_truncated_adaptive', 'tinygsm_sfm'])
def test_sfm_experiments_select_physical_adaptation(experiment):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        config = compose(config_name='config', overrides=[f'+experiment={experiment}'])
    assert config.noise.adaptive_coordinate == 'physical'
    assert not config.algo.invert_time_convention
    assert config.algo.auto_truncation is (
        experiment == 'sudoku_sfm_truncated_adaptive'
    )

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
    ("mdlm", ("sampling.predictor=ancestral_cache",)),
    ("d3pm", ("sampling.predictor=ancestral",)),
    ("sedd", ("sampling.predictor=analytic",)),
    ("cadd", ("sampling.predictor=cadd",)),
    ("candi", ("sampling.predictor=candi_cached",)),
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
    if entry[0] not in {"ar", "mdlm_perm"}
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
        "noise=log-linear-adaptive",
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

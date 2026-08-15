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
    ("candi_ra_change", ("sampling.predictor=candi",)),
    ("cont_diff_onehot", ("sampling.predictor=continuous_onehot",)),
    ("embedding_diffusion", ("sampling.predictor=continuous_embedding",)),
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
    (
        "distillation",
        (
            "sampling.predictor=ancestral",
            f"algo.integral_cache_path={CONFIG_DIR}/../integral/gpt2.pkl",
        ),
    ),
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

    assert len(model.backbone.blocks) == 1
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

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from algos import algo_registry
from dataloader import Text8Tokenizer


CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")
VARIANTS = (
    "replaid_keep_mse",
    "replaid_keep_hybrid_elbo",
    "replaid_keep_mse_reveal_ce",
)


def make_model(variant):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        config = compose(
            config_name="config",
            overrides=[
                f"algo={variant}",
                "model=tiny",
                "data=ptb",
                "training.ema=0",
                "model.length=4",
                "model.hidden_size=8",
                "model.cond_dim=8",
                "model.n_blocks=1",
                "model.n_heads=2",
                "model.dropout=0",
                "sampling.predictor=replaid_keep_ddpm",
            ],
        )
    return algo_registry.get(config.algo.name)(config, Text8Tokenizer())


@pytest.mark.parametrize("variant", VARIANTS)
def test_each_keep_objective_has_finite_loss_and_embedding_gradient(variant):
    torch.manual_seed(13)
    model = make_model(variant)
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    result = model._loss(tokens, torch.ones_like(tokens), train_mode=True)
    result.loss.backward()

    gradient = model.backbone.token_embedding.grad
    assert torch.isfinite(result.loss)
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_cadd_reverse_event_weights_form_a_probability_partition():
    model = make_model("replaid_keep_hybrid_elbo")
    gamma_t = torch.tensor([3.0, 0.5, -1.0])
    gamma_s = torch.tensor([2.0, 0.0, -2.0])

    rho_flip, rho_keep, mse_weight = model.cadd_step_weights(gamma_t, gamma_s)

    assert torch.allclose(rho_flip + rho_keep, torch.ones_like(rho_flip))
    assert torch.all((0 <= rho_flip) & (rho_flip <= 1))
    assert torch.all((0 <= rho_keep) & (rho_keep <= 1))
    assert torch.all(mse_weight > 0)


def test_keep_conditioning_uses_clean_embeddings_only_at_kept_positions(monkeypatch):
    model = make_model("replaid_keep_mse")
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    observed = {}
    original = model.denoise

    def capture(z_t, gamma, context):
        observed["input"] = z_t.detach().clone()
        observed["context"] = context.detach().clone()
        return original(z_t, gamma, context)

    monkeypatch.setattr(model, "denoise", capture)
    torch.manual_seed(19)
    model._loss(tokens, torch.ones_like(tokens), train_mode=True)
    clean = model.backbone.embed_tokens(tokens).detach()
    context = observed["context"]
    kept = context.square().sum(-1) > 0

    assert torch.allclose(context[kept], clean[kept])
    assert torch.count_nonzero(context[~kept]) == 0
    assert torch.allclose(observed["input"][kept], clean[kept])

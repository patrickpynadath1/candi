"""Detached feedback, sampler lifecycle, and flag-only MDLM integration."""

import copy
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest
import torch

from core.models.dit import DiT
from core.noise_schedules.log_linear import LogLinear
from core.samplers.ancestral import AncestralSampler
from core.samplers.entropy_bounded import EntropyBoundedSampler
from core.modifiers.self_conditioning.session import prediction_session
from core.training.module import TrainingModule


def config(source="posterior_mean", *overrides):
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        return compose(
            config_name="config",
            overrides=[
                "+experiment=smoke",
                "output_dir=/tmp/self_conditioning_test",
                "data.cache_dir=/tmp/self_conditioning_data",
                "algorithm.length=7",
                "self_conditioning.enabled=true",
                f"self_conditioning.source={source}",
                "self_conditioning.probability=1",
                *overrides,
            ],
        )


def algorithm(source="posterior_mean", *overrides):
    algo = instantiate(config(source, *overrides).algorithm)
    # The baseline zero output head otherwise hides all backbone gradients.
    torch.nn.init.normal_(algo.model.output_layer.linear.weight, std=0.1)
    return algo


def inputs(algo, batch=3, length=7):
    return torch.full((batch, length), algo.process.mask_index), torch.linspace(
        0.2, 0.8, batch
    )


def recorder(model):
    calls = []

    def record(module, args, kwargs, output):
        calls.append(
            (
                tuple(x.detach().clone() for x in args),
                kwargs,
                output,
                torch.is_grad_enabled(),
            )
        )

    return calls, model.register_forward_hook(record, with_kwargs=True)


@pytest.mark.parametrize("source", ["posterior_mean", "hidden"])
def test_bootstrap_same_input_detached_and_adapter_gradients(source):
    algo = algorithm(source)
    state, keep = inputs(algo)
    calls, handle = recorder(algo.model)
    prediction = algo.training_prediction(state, keep)
    handle.remove()
    assert len(calls) == 2
    for bootstrap, main in zip(calls[0][0], calls[1][0]):
        torch.testing.assert_close(bootstrap, main)
    assert not calls[0][3] and calls[1][3]
    feedback = calls[1][1]["self_condition"]
    assert not feedback.requires_grad and feedback.grad_fn is None
    assert not calls[0][2].logits.requires_grad
    if source == "hidden":
        expected = calls[0][2].hidden
        torch.testing.assert_close(feedback, expected)
    else:
        expected = (
            algo.predictor.training_posterior(calls[0][2].logits, state).exp()
            @ algo.model.vocab_embed.embedding
        )
        torch.testing.assert_close(feedback, expected)
    (-prediction[..., 1].mean()).backward()
    for parameter in [
        next(algo.model.conditioning_adapter.parameters()),
        algo.model.vocab_embed.embedding,
    ]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("source", ["posterior_mean", "hidden"])
def test_probability_selection_and_validation_bootstrap(source):
    algo = algorithm(source, "self_conditioning.probability=0.5")
    state, keep = inputs(algo, batch=12)
    torch.manual_seed(16)
    selected = torch.rand(len(state)) < 0.5
    calls, handle = recorder(algo.model)
    torch.manual_seed(16)
    algo.training_prediction(state, keep)
    assert len(calls) == 2
    torch.testing.assert_close(calls[0][0][0], state[selected])
    feedback = calls[1][1]["self_condition"]
    assert feedback[~selected].count_nonzero() == 0
    assert feedback[selected].count_nonzero() > 0
    calls.clear()
    algo.eval()
    algo.training_prediction(state, keep)
    assert len(calls) == 2 and calls[0][0][0].shape == state.shape
    calls.clear()
    algo.prediction.evaluation_probability = 0
    algo.training_prediction(state, keep)
    assert len(calls) == 1 and calls[0][1]["self_condition"] is None
    handle.remove()


@pytest.mark.parametrize("source", ["posterior_mean", "hidden"])
def test_sessions_unshaped_feedback_reset_and_nested_independence(source):
    algo = algorithm(source).eval()
    state, keep = inputs(algo)
    shaped = copy.deepcopy(algo)
    shaped.predictor.temperature = 0.01
    shaped.predictor.top_k = 1
    shaped.predictor.top_p = 0.1
    plain_session, shaped_session = prediction_session(algo), prediction_session(shaped)
    with torch.no_grad():
        plain = plain_session.sampling_prediction(state, keep)
        sharp = shaped_session.sampling_prediction(state, keep)
    assert not torch.allclose(plain, sharp)
    torch.testing.assert_close(
        plain_session.state.conditioning, shaped_session.state.conditioning
    )
    fresh = prediction_session(algo)
    assert fresh.state.conditioning is None
    saved = plain_session.state.conditioning.clone()
    with torch.no_grad():
        fresh.sampling_prediction(state, keep)
    torch.testing.assert_close(plain_session.state.conditioning, saved)
    assert not plain_session.state.conditioning.requires_grad
    # The public raw model endpoint stays stateless and tensor-valued.
    assert isinstance(algo.model_logits(state, keep), torch.Tensor)


@pytest.mark.parametrize("source", ["posterior_mean", "hidden"])
@pytest.mark.parametrize(
    "sampler", [AncestralSampler(steps=3), EntropyBoundedSampler(max_steps=8)]
)
def test_sampler_no_extra_nfe_prompt_preservation_and_reset(source, sampler):
    algo = algorithm(source)
    tokens = torch.ones(2, 7, dtype=torch.long)
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[:, :2] = True
    calls, handle = recorder(algo.model)
    for _ in range(2):
        calls.clear()
        result, metadata = algo.generate(sampler, tokens, mask)
        assert len(calls) == metadata["nfe"]
        assert calls[0][1]["self_condition"] is None
        assert all(call[1]["self_condition"] is not None for call in calls[1:])
        torch.testing.assert_close(result[mask], tokens[mask])
        assert not result.eq(algo.process.mask_index).any()
        assert algo.training
    handle.remove()


def test_sampler_exception_does_not_leak_conditioning():
    algo = algorithm()
    tokens = torch.ones(1, 7, dtype=torch.long)
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    calls, handle = recorder(algo.model)
    with pytest.raises(RuntimeError, match="exceeded max_steps"):
        algo.generate(EntropyBoundedSampler(entropy_bound=0, max_steps=1), tokens, mask)
    assert algo.training
    calls.clear()
    algo.generate(AncestralSampler(steps=2), tokens, mask)
    assert calls[0][1]["self_condition"] is None
    handle.remove()


def test_adaptive_schedule_observes_only_main_objective():
    class ObservingSchedule(LogLinear):
        loss_signal = "objective"

        def __init__(self):
            super().__init__()
            self.calls = []

        def observe(self, time, loss, step, training):
            self.calls.append((time, loss, step, training))

    algo = algorithm()
    algo.schedule = ObservingSchedule()
    calls, handle = recorder(algo.model)
    tokens = torch.ones(3, 7, dtype=torch.long)
    result = algo.loss(tokens, torch.ones_like(tokens), step=19, training=True)
    assert len(calls) == 2 and len(algo.schedule.calls) == 1
    assert algo.schedule.calls[0][2:] == (19, True)
    assert not algo.schedule.calls[0][1].requires_grad
    assert torch.isfinite(result["loss"])
    handle.remove()


@pytest.mark.parametrize("source", ["posterior_mean", "hidden"])
def test_checkpoint_optimizer_and_ema_include_adapter(source, tmp_path):
    cfg = config(source)
    module = TrainingModule(OmegaConf.to_container(cfg, resolve=True))
    projection = next(module.algorithm.model.conditioning_adapter.parameters())
    optimizer = module.configure_optimizers()["optimizer"]
    assert any(
        projection is parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    assert len(module.ema.shadow_params) == len(list(module.algorithm.parameters()))
    tokens = torch.ones(3, 7, dtype=torch.long)
    for _ in range(2):
        optimizer.zero_grad()
        module.algorithm.loss(tokens, torch.ones_like(tokens))["loss"].backward()
        optimizer.step()
    payload = {"state_dict": module.state_dict(), "optimizer": optimizer.state_dict()}
    module.on_save_checkpoint(payload)
    torch.save(payload, tmp_path / "checkpoint.pt")
    restored = TrainingModule(OmegaConf.to_container(cfg, resolve=True))
    saved = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    restored.load_state_dict(saved["state_dict"], strict=True)
    restored.on_load_checkpoint(saved)
    restored_optimizer = restored.configure_optimizers()["optimizer"]
    restored_optimizer.load_state_dict(saved["optimizer"])
    for candidate, opt in [(module, optimizer), (restored, restored_optimizer)]:
        torch.manual_seed(32)
        opt.zero_grad()
        candidate.algorithm.loss(tokens, torch.ones_like(tokens))["loss"].backward()
        opt.step()
    for left, right in zip(module.parameters(), restored.parameters()):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_disabled_preserves_old_checkpoint_keys_rng_and_prediction():
    torch.manual_seed(8)
    old = DiT(13, hidden_size=32, cond_dim=16, n_blocks=1, n_heads=4, dropout=0)
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(8)
    explicit = DiT(
        13,
        hidden_size=32,
        cond_dim=16,
        n_blocks=1,
        n_heads=4,
        dropout=0,
        self_conditioning=False,
    )
    assert torch.equal(old_rng, torch.get_rng_state())
    assert old.state_dict().keys() == explicit.state_dict().keys()
    assert not any("conditioning_adapter" in key for key in explicit.state_dict())
    explicit.load_state_dict(old.state_dict(), strict=True)
    algo = algorithm(
        "posterior_mean", "self_conditioning.enabled=false", "model.dropout=0.2"
    )
    state, keep = inputs(algo)
    # Warm the legacy TorchScript dropout kernel before comparing execution paths.
    # Its optimized graph may consume RNG even for p=0, so compare against the
    # direct baseline rather than assuming the model itself never draws randomness.
    for _ in range(3):
        algo.model_logits(state, keep)
    rng = torch.get_rng_state().clone()
    baseline = algo.predictor.training_posterior(algo.model_logits(state, keep), state)
    baseline_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    result = algo.training_prediction(state, keep)
    assert torch.equal(baseline_rng, torch.get_rng_state())
    torch.testing.assert_close(result, baseline, atol=0, rtol=0)
    assert prediction_session(algo) is algo
    assert (
        "self_condition"
        not in Path(__file__).parents[1].joinpath("algorithms/mdlm.py").read_text()
    )


@pytest.mark.parametrize("source", ["posterior_mean", "hidden"])
def test_cuda_mixed_precision_training_and_both_samplers(source):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    algo = algorithm(source, "algorithm.length=12").cuda()
    tokens = torch.ones(3, 12, dtype=torch.long, device="cuda")
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[:, :3] = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = algo.loss(tokens, ~mask)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(next(algo.model.conditioning_adapter.parameters()).grad).all()
    for sampler in [AncestralSampler(steps=3), EntropyBoundedSampler(max_steps=12)]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result, _ = algo.generate(sampler, tokens, mask)
        torch.testing.assert_close(result[mask], tokens[mask])


def test_mismatched_architecture_fails_at_construction():
    with pytest.raises(Exception, match="enable self-conditioning together"):
        instantiate(config("posterior_mean", "model.self_conditioning=false").algorithm)


def test_hidden_source_does_not_request_embedding_table():
    algo = algorithm("hidden")

    def unavailable():
        raise AssertionError("hidden feedback must not read the token embedding table")

    algo.prediction_embedding_table = unavailable
    state, keep = inputs(algo)
    assert algo.training_prediction(state, keep).shape == (*state.shape, 13)


def test_time_conditioned_mdlm_uses_same_bootstrap_coordinate():
    algo = algorithm(
        "posterior_mean", "model=dit_conditioned", "algorithm.time_conditioning=true"
    )
    state, keep = inputs(algo)
    calls, handle = recorder(algo.model)
    algo.training_prediction(state, keep)
    assert len(calls) == 2
    torch.testing.assert_close(calls[0][0][1], -keep.log())
    torch.testing.assert_close(calls[1][0][1], -keep.log())
    handle.remove()


@pytest.mark.parametrize(
    "experiment",
    [
        "autoregressive",
        "sedd_absorb",
        "d3pm_absorb",
    ],
)
def test_unsupported_method_rejects_requested_flag(experiment):
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                f"+experiment={experiment}",
                "self_conditioning.enabled=true",
                "model.hidden_size=32",
                "model.n_blocks=1",
                "model.n_heads=4",
                "output_dir=/tmp/self_conditioning_test",
                "data.cache_dir=/tmp/self_conditioning_data",
            ],
        )
    with pytest.raises(Exception, match="self-conditioning"):
        TrainingModule(OmegaConf.to_container(cfg, resolve=True))


def continuous_algorithm(experiment, source, device="cpu"):
    overrides = [
        f"+experiment={experiment}",
        "self_conditioning.enabled=true",
        f"self_conditioning.source={source}",
        "self_conditioning.probability=1",
        "model.hidden_size=32",
        "model.cond_dim=16",
        "model.n_blocks=1",
        "model.n_heads=4",
        "model.dropout=0",
        "algorithm.length=7",
        "output_dir=/tmp/self_conditioning_test",
        "data.cache_dir=/tmp/self_conditioning_data",
    ]
    if experiment != "gaussian_diffusion_sudoku":
        overrides.append("representation.dimension=8")
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).parents[2] / "configs")
    ):
        cfg = compose(config_name="config", overrides=overrides)
    algo = instantiate(cfg.algorithm).to(device)
    if hasattr(algo.model, "output_layer"):
        torch.nn.init.normal_(algo.model.output_layer.linear.weight, std=0.1)
    return algo


@pytest.mark.parametrize("source", ["posterior_mean", "hidden"])
@pytest.mark.parametrize(
    "experiment",
    [
        "gaussian_diffusion_sudoku",
        "gaussian_diffusion_embedding_sudoku",
        "spherical_flow_matching",
    ],
)
def test_continuous_sources_train_and_sampler_feedback(experiment, source):
    from core.samplers.euler import EulerSampler
    from core.samplers.heun import HeunSampler
    from core.samplers.spherical import SphericalFlowSampler

    algo = continuous_algorithm(experiment, source)
    tokens = torch.ones(2, 7, dtype=torch.long)
    prompt = torch.zeros_like(tokens, dtype=torch.bool)
    prompt[:, :2] = True
    calls, handle = recorder(algo.model)
    loss = algo.loss(tokens, ~prompt)["loss"]
    assert len(calls) == 2
    for bootstrap, main in zip(calls[0][0], calls[1][0]):
        torch.testing.assert_close(bootstrap, main)
    feedback = calls[1][1]["self_condition"]
    expected_width = 32 if source == "hidden" else algo.representation.spec.dimension
    assert feedback.shape == (2, 7, expected_width)
    assert not feedback.requires_grad
    if source == "posterior_mean":
        probs = algo.predictor.training_posterior(
            algo.prediction_logits(calls[0][2].logits), calls[0][0][0]
        ).exp()
        torch.testing.assert_close(feedback, algo.representation.posterior_mean(probs))
        if experiment == "spherical_flow_matching":
            assert (feedback.norm(dim=-1) < 0.99).any(), (
                "posterior mean must not be sphere-normalized"
            )
    loss.backward()
    gradient = next(algo.model.conditioning_adapter.parameters()).grad
    assert (
        gradient is not None
        and torch.isfinite(gradient).all()
        and gradient.abs().sum() > 0
    )
    samplers = (
        [SphericalFlowSampler(steps=3)]
        if experiment == "spherical_flow_matching"
        else [EulerSampler(steps=2), HeunSampler(steps=2)]
    )
    for sampler in samplers:
        calls.clear()
        generated, metadata = algo.generate(sampler, tokens, prompt)
        assert len(calls) == metadata["nfe"]
        torch.testing.assert_close(generated[prompt], tokens[prompt])
        assert calls[0][1]["self_condition"] is None
        assert calls[-1][1]["self_condition"] is not None
        if isinstance(sampler, HeunSampler):
            assert len(calls) == 5
            assert calls[1][1]["self_condition"] is None
            torch.testing.assert_close(
                calls[2][1]["self_condition"], calls[3][1]["self_condition"]
            )
            if source == "hidden":
                torch.testing.assert_close(
                    calls[2][1]["self_condition"],
                    calls[1][2].hidden,
                )
    handle.remove()


def test_staged_session_rollback_and_adaptive_ode_guard():
    from core.samplers.ode import ODESampler

    algo = continuous_algorithm("gaussian_diffusion_sudoku", "hidden").eval()
    session = prediction_session(algo)
    state = torch.randn(2, 7, 12)
    time = torch.tensor([0.5, 0.5])
    with torch.no_grad():
        session.sampling_prediction(state, time, algo.schedule(time))
        previous = session.state.conditioning.clone()
        with pytest.raises(RuntimeError, match="rejected"):
            with session.step():
                session.sampling_prediction(state + 1, time, algo.schedule(time))
                assert session.state.pending is not None
                torch.testing.assert_close(session.state.conditioning, previous)
                raise RuntimeError("rejected")
        torch.testing.assert_close(session.state.conditioning, previous)
        assert session.state.pending is None and not session.state.staged
    tokens = torch.ones(2, 7, dtype=torch.long)
    with pytest.raises(ValueError, match="adaptive ODE"):
        algo.generate(ODESampler(), tokens, torch.zeros_like(tokens, dtype=torch.bool))


@pytest.mark.parametrize("source", ["posterior_mean", "hidden"])
@pytest.mark.parametrize(
    "experiment",
    [
        "gaussian_diffusion_sudoku",
        "gaussian_diffusion_embedding_sudoku",
        "spherical_flow_matching",
    ],
)
def test_continuous_cuda_mixed_precision(experiment, source):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from core.samplers.heun import HeunSampler
    from core.samplers.spherical import SphericalFlowSampler

    algo = continuous_algorithm(experiment, source, "cuda")
    tokens = torch.ones(2, 7, dtype=torch.long, device="cuda")
    prompt = torch.zeros_like(tokens, dtype=torch.bool)
    prompt[:, :2] = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = algo.loss(tokens, ~prompt)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    sampler = (
        SphericalFlowSampler(steps=3)
        if experiment == "spherical_flow_matching"
        else HeunSampler(steps=2)
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        generated, metadata = algo.generate(sampler, tokens, prompt)
    torch.testing.assert_close(generated[prompt], tokens[prompt])
    assert metadata["nfe"] == (3 if experiment == "spherical_flow_matching" else 5)


def test_mean_requires_probability_capability_hidden_does_not():
    from core.modifiers.self_conditioning.prediction import PredictionCoordinator

    class RawPrediction:
        def training_posterior(self, logits, observed):
            return logits

    model = DiT(
        13,
        hidden_size=32,
        cond_dim=16,
        n_blocks=1,
        n_heads=4,
        dropout=0,
        self_conditioning=True,
        self_conditioning_source="hidden",
    )
    predictor = RawPrediction()
    with pytest.raises(TypeError, match="posterior_mean.*RawPrediction.*source=hidden"):
        PredictionCoordinator(enabled=True).validate(model, predictor)
    hidden = PredictionCoordinator(enabled=True, source="hidden", probability=1)
    hidden.validate(model, predictor)
    state = torch.ones(2, 7, dtype=torch.long)
    output = hidden.predict(
        model, predictor, (state, torch.zeros(2)), state, "training_posterior"
    )
    assert output.shape == (2, 7, 13)


@pytest.mark.parametrize("source", ["hidden", "posterior_mean"])
def test_feedback_initially_zero_and_can_learn(source):
    from core.models.prediction import make_conditioning

    adapter = make_conditioning(source, 32, 32)
    hidden = torch.randn(2, 7, 32)
    feedback = torch.randn_like(hidden) * 100
    torch.testing.assert_close(adapter(hidden, feedback), hidden, rtol=0, atol=0)
    torch.testing.assert_close(adapter(hidden, None), hidden, rtol=0, atol=0)
    adapter(hidden, feedback).square().sum().backward()
    assert next(adapter.parameters()).grad.abs().sum() > 0
    if source == "hidden":
        assert not any(
            isinstance(module, torch.nn.Linear) for module in adapter.modules()
        )
        assert adapter.norm.weight.count_nonzero() == 0
        assert adapter.norm.bias.count_nonzero() == 0
        with torch.no_grad():
            adapter.norm.weight.fill_(0.5)
            adapter.norm.bias.fill_(0.1)
        torch.testing.assert_close(
            adapter(hidden, feedback), hidden + adapter.norm(feedback)
        )
        torch.testing.assert_close(adapter(hidden, None), hidden + 0.1)


def test_default_bootstrap_probability():
    from core.modifiers.self_conditioning.prediction import PredictionCoordinator

    assert PredictionCoordinator().probability == 0.9
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[2] / "configs"),
        version_base=None,
    ):
        cfg = compose(config_name="config")
    assert cfg.self_conditioning.probability == 0.9


def test_entropy_budget_readout_preserves_prompts_and_counts_actual_calls():
    algo = algorithm("hidden")
    tokens = torch.ones(3, 7, dtype=torch.long)
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[0, :] = True
    mask[1, :6] = True
    mask[2, :2] = True
    calls, handle = recorder(algo.model)
    generated, metadata = algo.generate(
        EntropyBoundedSampler(entropy_bound=0, max_steps=1, final_denoise="greedy"),
        tokens,
        mask,
    )
    assert metadata["nfe"] == len(calls) == 2
    assert metadata["nfe_per_sample"] == [0, 1, 2]
    torch.testing.assert_close(generated[mask], tokens[mask])
    assert not generated.eq(algo.process.mask_index).any()
    assert calls[1][1]["self_condition"] is not None
    handle.remove()

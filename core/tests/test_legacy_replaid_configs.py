"""Configs and checkpoints saved by the removed RePlaid / RePlaidHybrid classes.

The fixtures follow the structure of real saved run configs: the old classes took
the clock, corruption and context scale as constructor arguments, which now live
in the preset's views, time sampler and model input."""
import copy

import pytest
import torch
from omegaconf import OmegaConf

from core.sweeps.configuration import compose
from core.training.checkpoints import load_weights
from core.training.legacy_checkpoints import (
    legacy_recipe,
    migrate_legacy_config,
    migrate_legacy_config_dict,
)
from core.training.module import TrainingModule

TINY = dict(vocab_size=5, state_dim=4, hidden_size=32, cond_dim=16, n_blocks=1, n_heads=4)


def _model(self_conditioning, epsilon):
    return {
        "_target_": "core.models.modular.ModularDiT",
        "input": {"_target_": "core.models.modular.LinearInput", "_partial_": True,
                  "bias": False, "preconditioning": "unit_embedding"},
        "output": {"_target_": "core.models.modular.AdaLNHead", "_partial_": True,
                   "adaLN": True, "temperature": False},
        **TINY, "dropout": 0.0, "autocast_cache": False,
        "time": {"_target_": "core.models.modular.TimeConditioning", "_partial_": True,
                 "coordinate": "negative_log_snr", "epsilon": epsilon, "input_projection": True},
        "self_conditioning": {"_target_": "core.models.modular.SelfConditioning", "_partial_": True,
                              "enabled": self_conditioning, "source": "posterior_mean",
                              "space": "state", "initialization": "default" if self_conditioning else "zero",
                              "scale": "unit_embedding" if self_conditioning else 1.0},
    }


PREDICTOR = {"_target_": "core.parameterizations.gaussian_prior.GaussianPriorPosterior",
             "output_prior": True, "temperature": 1.0, "softcap": None, "top_k": 0, "top_p": 1.0}
REPRESENTATION = {"_target_": "core.representations.embedding.EmbeddingRepresentation",
                  "vocab_size": 5, "dimension": 4, "unit_normalized": True,
                  "trainable": True, "project_after_update": True}


def _saved(preset, algorithm, sampler):
    """A resolved run config whose recipe sections use the old class layout."""
    base = OmegaConf.to_container(
        compose(preset, ["task.vocab_size=5", "task.length=4", "training.ema=0.9"]),
        resolve=True,
    )
    base["algorithm"] = algorithm
    base["model"] = copy.deepcopy(algorithm["model"])
    base["sampler"] = sampler
    base["compile"] = False
    return base


def legacy_hybrid():
    algorithm = {
        "_target_": "core.algorithms.replaid_hybrid.RePlaidHybrid",
        "model": _model(False, 1e-30), "predictor": PREDICTOR,
        "process": {"_target_": "core.processes.gaussian.GaussianProcess"},
        "clock": {"_target_": "core.noise_schedules.shared_clock.PiecewiseInverseCDFClock",
                  "lower": -11.0, "upper": 4.0, "bins": 8, "uniform_mix": 0.05},
        "representation": REPRESENTATION,
        "objective": {"_target_": "core.objectives.hybrid_elbo.HybridELBO"},
        "time_sampler": None, "length": 4,
        "prediction": {"_target_": "core.modifiers.self_conditioning.prediction.PredictionCoordinator",
                       "enabled": False, "source": "posterior_mean", "probability": 0.9,
                       "evaluation_probability": 1.0, "gradient_mode": "detached"},
        "corruption": {"_target_": "core.processes.forced_mask.ForcedTargetMask"},
        "context_scale": 1.5,
    }
    sampler = {"_target_": "core.samplers.composed.ComposedSampler", "steps": 3, "epsilon": 0.0,
               "prior_mode": "standard_normal", "grid_precision": "float64",
               "synchronize_committed": True, "final_denoise": "state",
               "transition": {"_target_": "core.samplers.policies.ReverseTransition",
                              "final_transition_noise": True}}
    return _saved("replaid_hybrid", algorithm, sampler)


def legacy_replaid():
    algorithm = {
        "_target_": "core.algorithms.replaid.RePlaid",
        "model": _model(True, 1e-6), "predictor": PREDICTOR,
        "process": {"_target_": "core.processes.gaussian.GaussianProcess"},
        "schedule": {"_target_": "core.noise_schedules.learned_log_snr.PlaidGammaSchedule",
                     "gamma_min": -12.0, "gamma_max": 4.5, "hidden_dim": 8},
        "time_sampler": {"_target_": "core.time_sampling.adaptive_allocation.AdaptiveTimeAllocation",
                         "momentum": 0.9},
        "objective": {"_target_": "core.objectives.replaid.RePlaidObjective"},
        "length": 4, "representation": REPRESENTATION,
        "prediction": {"_target_": "core.modifiers.self_conditioning.prediction.PredictionCoordinator",
                       "enabled": True, "source": "posterior_mean", "probability": 0.25,
                       "evaluation_probability": 1.0, "gradient_mode": "detached",
                       "selection": "fixed_count"},
        "corrupt_all": False,
    }
    sampler = {"_target_": "core.samplers.gaussian_ddpm.GaussianDDPMSampler", "steps": 3,
               "epsilon": 0.0, "prior_mode": "standard_normal", "prompt_mode": "clean",
               "self_conditioning_mode": "bootstrap", "self_conditioning": True,
               "final_transition_noise": True}
    return _saved("replaid", algorithm, sampler)


def test_non_legacy_configs_pass_through_untouched():
    config = compose("replaid_hybrid")
    assert legacy_recipe(config) is None
    assert migrate_legacy_config(config) is config


def test_hybrid_config_keeps_the_saved_recipe_values():
    module = TrainingModule(migrate_legacy_config(OmegaConf.create(legacy_hybrid())))
    algorithm = module.algorithm
    content = algorithm.model.input.views[0]
    assert type(algorithm).__name__ == "HybridDiffusion"
    assert algorithm.model.input.context_projection.weight.shape == (32, 4)
    assert algorithm.model.input.context_scale == 1.5
    gamma_max = content.schedule.gamma_min + torch.nn.functional.softplus(content.schedule.gamma_delta_raw)
    assert float(content.schedule.gamma_min) == -11.0
    assert float(gamma_max) == pytest.approx(4.0)
    assert algorithm.time_sampler.clock_logits.shape == (8,)
    assert type(algorithm.model.input.views[1].process.proposal).__name__ == "ForcedTargetMask"
    assert migrate_legacy_config(OmegaConf.create(legacy_hybrid())).sampler.steps == 3


def test_replaid_config_keeps_prompt_policy_and_schedule():
    config = migrate_legacy_config(OmegaConf.create(legacy_replaid()))
    module = TrainingModule(config)
    assert type(module.algorithm).__name__ == "GaussianDiffusion"
    assert module.algorithm.corrupt_all is False
    assert config.sampler.prompt_mode == "clean" and config.sampler.steps == 3
    assert module.algorithm.own_views[0].schedule.input_weight.shape == (8,)
    assert module.algorithm.time_sampler.momentum == 0.9


def _legacy_hybrid_state(module):
    """Write the module's state the way RePlaidHybrid did: shared clock under both views."""
    state = {}
    for key, value in module.state_dict().items():
        renames = {
            "algorithm.model.input.views.0.schedule.gamma_min": "gamma_min",
            "algorithm.model.input.views.0.schedule.gamma_delta_raw": "width_raw",
            "algorithm.time_sampler.clock_logits": "clock_logits",
        }
        if key in renames:
            for view in (0, 1):
                state[f"algorithm.model.input.views.{view}.schedule.clock.{renames[key]}"] = value
        else:
            state[key] = value
    return state


def test_legacy_hybrid_checkpoint_loads_into_the_rebuilt_module(tmp_path):
    saved = legacy_hybrid()
    config = migrate_legacy_config(OmegaConf.create(saved))
    torch.manual_seed(0)
    source = TrainingModule(config)
    with torch.no_grad():
        for parameter in source.algorithm.parameters():
            parameter.normal_()
    names = [name for name, _ in source.algorithm.named_parameters()]
    # The old parameter order put the bin logits right after the clock endpoints.
    old = [n for n in names if n != "time_sampler.clock_logits"]
    old.insert(old.index("model.input.views.0.schedule.gamma_delta_raw") + 1, "time_sampler.clock_logits")
    shadows = {n: torch.full_like(p, float(i)) for i, (n, p) in enumerate(source.algorithm.named_parameters())}
    checkpoint = {
        "core_format_version": 1,
        "global_step": 7,
        "state_dict": _legacy_hybrid_state(source),
        "hyper_parameters": {"config": saved},
        "ema": {"decay": 0.9, "num_updates": 7, "shadow_params": [shadows[n] for n in old]},
    }
    path = tmp_path / "legacy.ckpt"
    torch.save(checkpoint, path)

    target = TrainingModule(migrate_legacy_config(OmegaConf.create(saved)))
    assert load_weights(target, str(path)) == 7  # strict load, recipe check passes
    for (name, expected), actual in zip(source.algorithm.named_parameters(), target.algorithm.parameters()):
        torch.testing.assert_close(actual, expected, msg=name)
    for name, shadow in zip(names, target.ema.shadow_params):
        torch.testing.assert_close(shadow, shadows[name], msg=name)


def test_saved_config_dict_migration_is_idempotent():
    once = migrate_legacy_config_dict(legacy_replaid())
    assert migrate_legacy_config_dict(once) is once

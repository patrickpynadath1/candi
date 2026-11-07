"""Evaluation must not silently pair weights with a different parameter-free recipe."""

from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from core.training.checkpoints import check_recipe, recipe_mismatches


def recipe(**changes):
    algorithm = {
        "_target_": "core.algorithms.mdlm.MDLM",
        "schedule": {"_target_": "LogLinear", "epsilon": 1e-3},
        "predictor": {"temperature": 1.0, "top_k": 0, "mask_index": 11},
        "time_sampler": {"antithetic": True},
        "time_conditioning": False,
    }
    config = OmegaConf.create(algorithm)
    for key, value in changes.items():
        OmegaConf.update(config, key, value)
    return {"algorithm": OmegaConf.to_container(config), "sampler": {"steps": 8}}


def test_inference_knobs_may_differ():
    saved = recipe()
    current = recipe(**{"predictor.temperature": 0.5, "time_sampler.antithetic": False})
    current["sampler"] = {"steps": 64}
    assert recipe_mismatches(saved, current) == []


def test_parameter_free_recipe_changes_are_reported():
    saved = recipe()
    current = recipe(**{"schedule.epsilon": 0.0, "time_conditioning": True})
    assert recipe_mismatches(saved, current) == [
        "algorithm.schedule.epsilon",
        "algorithm.time_conditioning",
    ]
    module = SimpleNamespace(config=OmegaConf.create(current))
    checkpoint = {"hyper_parameters": {"config": saved}}
    with pytest.raises(ValueError, match="different recipe"):
        check_recipe(module, checkpoint)
    assert check_recipe(module, checkpoint, allow_mismatch=True)
    assert check_recipe(module, {}) == []

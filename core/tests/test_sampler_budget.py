"""Budget allocation through the tuning worker, measured by actual sampling."""

import json

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from core.samplers.composed import ComposedSampler
from core.samplers.correctors import SphericalLangevinCorrector
from core.sweeps import nested_tuning as tuning
from core.tests.test_sampler_equivalence import build_algorithm


@pytest.mark.parametrize("corrections,expected_nfe", [(0, 8), (1, 7), (2, 7)])
def test_tuning_budget_includes_tuned_corrections(
    tmp_path, monkeypatch, corrections, expected_nfe
):
    from core.training import runner

    algo = build_algorithm("vmf", "uniform", "none")
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create({"evaluation": {}}), config_path)
    observed = []

    def prepare(payload, destination, resources):
        destination.mkdir(parents=True)
        config = OmegaConf.create(payload["config"])
        config.output = str(destination)
        return config

    def run(config):
        sampler = instantiate(config.sampler)
        tokens = torch.tensor([[1, 2, 3, 4]])
        mask = torch.tensor([[True, False, False, False]])
        output, info = sampler.sample(algo, tokens, mask)
        assert output[0, 0] == tokens[0, 0]
        observed.append(info["nfe"])
        from pathlib import Path

        (Path(config.output) / "results.json").write_text(
            json.dumps(
                {
                    "max_nfe": info["nfe"],
                    "accuracy": 1.0,
                }
            )
        )

    monkeypatch.setattr(tuning, "_prepare", prepare)
    monkeypatch.setattr(runner, "run", run)
    corrector = {
        "_target_": "core.samplers.correctors.SphericalLangevinCorrector",
        "steps": 1,
    }
    tuning._evaluate_worker(
        {
            "checkpoint": {"config": str(config_path), "checkpoint": "unused.ckpt"},
            "evaluation": {
                "nfe": 8,
                "seeds": [42],
                "indices": [0],
                "score": {"metric": "accuracy"},
            },
            "policy": {
                "budget_parameter": "steps",
                "sampler": {
                    "_target_": "core.samplers.composed.ComposedSampler",
                    "corrector": corrector if corrections else None,
                },
            },
            "parameters": {"sampler.corrector.steps": corrections}
            if corrections
            else {},
            "resources": {"gpu": 0},
        },
        tmp_path / "evaluation",
    )
    assert observed == [expected_nfe]


def test_128_budget_and_insufficient_budget():
    sampler = ComposedSampler(corrector=SphericalLangevinCorrector())
    assert sampler.steps_for_nfe(128) == 63
    with pytest.raises(ValueError, match="cannot fit"):
        sampler.steps_for_nfe(2)
    with pytest.raises(ValueError, match="integer"):
        sampler.steps_for_nfe(3.5)


@pytest.mark.parametrize("discrete", ["uniform", "masked"])
def test_correction_preserves_discrete_view_and_prompts(discrete):
    algo = build_algorithm("vmf", discrete, "posterior_mean")
    corrector = SphericalLangevinCorrector()
    corrector.validate(algo.views)
    discrete_view = algo.views[1]
    state = torch.tensor([[1, 2, 3]])
    assert corrector(discrete_view, state, None, None) is state
    tokens = torch.tensor([[1, 2, 3, 4]])
    mask = torch.tensor([[True, False, False, False]])
    output, info = ComposedSampler(steps=2, corrector=corrector).sample(
        algo, tokens, mask
    )
    assert output[0, 0] == tokens[0, 0]
    assert info["nfe"] == 5


def test_correction_rejects_no_compatible_view():
    algo = build_algorithm("gaussian", "uniform", "none")
    with pytest.raises(ValueError, match="spherical view with a score"):
        SphericalLangevinCorrector().validate(algo.views)

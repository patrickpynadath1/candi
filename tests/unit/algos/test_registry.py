from pathlib import Path

import pytest
from catalogue import RegistryError
from omegaconf import OmegaConf

from algos import algo_registry
from models import model_registry


CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "algo"


@pytest.mark.parametrize("config_path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda path: path.stem)
def test_algorithm_configs_resolve_registered_algorithm_and_backbone(config_path):
    config = OmegaConf.load(config_path)
    assert algo_registry.get(config.name) is not None
    assert model_registry.get(config.backbone) is not None


@pytest.mark.parametrize(
    ("legacy_name", "canonical_name"),
    [("d3pm_absorb", "d3pm"), ("sedd_absorb", "sedd")],
)
def test_legacy_algorithm_names_resolve_to_canonical_class(
    legacy_name, canonical_name
):
    assert algo_registry.get(legacy_name) is algo_registry.get(canonical_name)


def test_unknown_algorithm_raises_descriptive_error():
    with pytest.raises(RegistryError, match="unknown_algorithm"):
        algo_registry.get("unknown_algorithm")

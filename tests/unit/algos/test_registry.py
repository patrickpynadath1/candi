from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

import main  # noqa: F401 -- registers config resolvers

from algos import algo_registry
from models import model_registry


CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"
CONFIG_DIR = CONFIG_ROOT / "algo"


@pytest.mark.parametrize("config_path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda path: path.stem)
def test_algorithm_configs_resolve_registered_algorithm_and_backbone(config_path):
    # Compose through Hydra so config-group defaults/inheritance are tested.
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        config = compose(config_name="config", overrides=[f"algo={config_path.stem}"])
    assert algo_registry.get(config.algo.name) is not None
    assert model_registry.get(config.algo.backbone) is not None


@pytest.mark.parametrize(
    ("legacy_name", "canonical_name"),
    [("d3pm_absorb", "d3pm"), ("sedd_absorb", "sedd")],
)
def test_legacy_algorithm_names_resolve_to_canonical_class(
    legacy_name, canonical_name
):
    assert algo_registry.get(legacy_name) is algo_registry.get(canonical_name)

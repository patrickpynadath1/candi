"""Evaluate pretrained frontier grids: uv run python frontier.py [overrides]."""

from pathlib import Path
import hydra
from core.sweeps.frontier import run


@hydra.main(version_base=None, config_path="configs", config_name="frontier")
def main(config):
    run(config, evaluation_entrypoint=Path(__file__).with_name("main.py"))


if __name__ == "__main__":
    main()

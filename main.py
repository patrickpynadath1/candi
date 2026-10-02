"""Train, evaluate, or sample: uv run python main.py [Hydra overrides]."""

import hydra
from core.training.runner import run


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    run(config)


if __name__ == "__main__":
    main()

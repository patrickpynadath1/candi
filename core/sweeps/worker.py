"""Fresh-process entrypoint for an already resolved sweep configuration."""

import sys

from omegaconf import OmegaConf


def main():
    from core.training.runner import run

    run(OmegaConf.load(sys.argv[1]))


if __name__ == "__main__":
    main()

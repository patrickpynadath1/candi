"""Torchrun entrypoint for a resolved Tune trial; never initializes Ray."""

import sys

from omegaconf import OmegaConf

from core.training.runner import run


if __name__ == "__main__":
    run(OmegaConf.load(sys.argv[1]))

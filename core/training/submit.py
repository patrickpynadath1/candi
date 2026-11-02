"""Submit a resolved training configuration to a labeled Ray worker."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--address", required=True)
    parser.add_argument("--working-dir", type=Path, required=True)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--cpus", type=int, default=4)
    args = parser.parse_args()
    import ray
    from core.training.runner import run

    ray.init(
        address=args.address,
        runtime_env={"working_dir": str(args.working_dir.resolve())},
    )
    config = OmegaConf.load(args.config)
    worker = ray.remote(
        num_cpus=args.cpus,
        num_gpus=1,
        max_calls=1,
        label_selector={"queue": args.queue},
    )(run)
    try:
        ray.get(worker.remote(config))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

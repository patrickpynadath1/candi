"""Train and evaluate masked/Gaussian Text8 block insertion with self-conditioning."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import sys

from text8_insertion_sc_experiments import trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--gpus", type=int, nargs=2, default=[4, 5])
    args = parser.parse_args()
    if len(set(args.gpus)) != 2:
        parser.error("assign a distinct GPU to each experiment")
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [
            pool.submit(
                trial,
                kind,
                gpu,
                args.cache_dir,
                experiment_prefix="text8_block_insertion",
            )
            for kind, gpu in zip(("masked", "gaussian"), args.gpus)
        ]
        outcomes = [job.result() for job in jobs]
    return 0 if all(outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())

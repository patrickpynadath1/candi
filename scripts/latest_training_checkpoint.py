"""Select the most advanced readable checkpoint after a Slurm requeue."""

import argparse
import pickle
from pathlib import Path
import sys

import torch


def latest_checkpoint(save_dir, fallback=None):
    candidates = set((Path(save_dir) / "checkpoints").glob("*.ckpt"))
    if fallback:
        candidates.add(Path(fallback))
    readable = []
    for path in sorted(candidates):
        try:
            # These are this run's own trusted Lightning checkpoints. mmap
            # avoids reading all model/optimizer tensors just to inspect step.
            state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            if not state.get("optimizer_states"):
                raise ValueError("missing optimizer state")
            step = int(state["global_step"])
            del state
            readable.append((step, path.name == "last.ckpt", str(path)))
        except (OSError, RuntimeError, ValueError, KeyError, EOFError, pickle.UnpicklingError) as error:
            print(f"Skipping unusable checkpoint {path}: {error}", file=sys.stderr)
    if not readable:
        raise RuntimeError(f"No resumable checkpoint found in {save_dir}")
    return max(readable)[2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("save_dir")
    parser.add_argument("--fallback")
    args = parser.parse_args()
    print(latest_checkpoint(args.save_dir, args.fallback))

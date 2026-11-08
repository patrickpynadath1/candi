"""Convert a checkpoint saved before the view-path unification.

  python scripts/convert_checkpoint.py IN OUT --config RUN_CONFIG

RUN_CONFIG is the run's resolved config (its saved ``config.yaml``). OUT then
loads strictly into today's TrainingModule for that config, with EMA (and,
when possible, optimizer state) reordered to today's parameter order. IN is
never modified or overwritten. See core/models/legacy_checkpoints.py.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.input.resolve():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    import torch
    from omegaconf import OmegaConf

    from core.models.legacy_checkpoints import convert_checkpoint

    # Checkpoints are trusted local training artifacts.
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    if checkpoint.get("core_format_version") != 1:
        raise ValueError("not a core checkpoint; use evaluation.legacy=true/candi imports")
    converted = convert_checkpoint(checkpoint, OmegaConf.load(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    torch.save(converted, temporary)
    temporary.rename(args.output)
    print(json.dumps({"output": str(args.output), **converted["conversion"]}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Materialize the exact GSM8K indices used by the local 64-sample run."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--local-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.source.read_text())
    results = json.loads(args.local_results.read_text())
    indices = [record["dataset_index"] for record in results["records"]]
    args.output.write_text(json.dumps([source[index] for index in indices], indent=2))
    print(f"wrote {len(indices)} examples to {args.output}")


if __name__ == "__main__":
    main()

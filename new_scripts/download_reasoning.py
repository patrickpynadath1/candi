#!/usr/bin/env python3
"""Download public source archives. Conversion manifests pin the actual bytes."""

import argparse
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasks.reasoning_data import DRIVE

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=Path("data_cache/reasoning_raw"))
    p.add_argument("--source", choices=["eqr", "shah", "all"], default="all")
    p.add_argument(
        "--revision",
        default="main",
        help="EqR dataset revision; resolves to a pinned snapshot",
    )
    args = p.parse_args()
    if args.source in ("eqr", "all"):
        from huggingface_hub import snapshot_download

        snapshot_download(
            "locuslab/EqR-data",
            repo_type="dataset",
            revision=args.revision,
            local_dir=args.raw / "eqr",
            allow_patterns=["maze-30x30-unique-1k/**", "sudoku-extreme-1k-aug-1000/**"],
        )
    if args.source in ("shah", "all"):
        directory = args.raw / "shah"
        directory.mkdir(parents=True, exist_ok=True)
        for name, identifier in DRIVE.items():
            target = directory / name
            if target.exists():
                print(f"already downloaded: {target}")
                continue
            url = f"https://drive.usercontent.google.com/download?id={identifier}&export=download&confirm=t"
            with urllib.request.urlopen(url, timeout=120) as response:
                if "text/html" in response.headers.get("Content-Type", ""):
                    raise RuntimeError("download returned an HTML page")
                temporary = target.with_suffix(target.suffix + ".part")
                with temporary.open("wb") as stream:
                    for chunk in iter(lambda: response.read(8 << 20), b""):
                        stream.write(chunk)
                temporary.rename(target)
            print(target)

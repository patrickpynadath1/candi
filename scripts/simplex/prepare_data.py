"""Audit and convert cached puzzles to SDM J.4 without regenerating full grids."""

import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tasks.sudoku_generator import count_solutions


def unique(puzzle):
    return count_solutions(puzzle.reshape(9, 9).tolist()) == 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--project", default="simplex-sudoku")
    args = parser.parse_args()
    destination = (
        args.cache_dir / "sudoku_hard_train200000_valid5000_seed42_paper14_exactclues"
    )
    if destination.exists():
        raise FileExistsError(f"Refusing to replace {destination}")
    with wandb.init(
        project=args.project,
        job_type="data-preparation",
        name="audit-paper-sudoku",
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        mode="online",
    ) as run:
        source = load_from_disk(str(args.source))
        tokens = np.concatenate(
            [
                np.array(source["train"][:200000]["input_ids"], dtype=np.int64),
                np.array(source["validation"][:5000]["input_ids"], dtype=np.int64),
            ]
        )
        if tokens.shape != (205000, 180):
            raise ValueError(
                "Expected 200k train and 5k validation puzzles of length 180"
            )
        positions = np.array([1 + r * 10 + c for r in range(9) for c in range(9)])
        puzzles, solutions = tokens[:, positions], tokens[:, positions + 90]
        assert np.all((puzzles >= 0) & (puzzles <= 9)) and np.all(
            (solutions >= 1) & (solutions <= 9)
        )
        assert np.all((puzzles != 0).sum(1) == 30)
        assert np.all((puzzles == 0) | (puzzles == solutions))
        assert np.all(tokens[:, [0, 90]] == 11)
        separators = [10 * r for r in range(1, 9)]
        assert np.all(tokens[:, separators] == 10) and np.all(
            tokens[:, np.array(separators) + 90] == 10
        )
        grids = solutions.reshape(-1, 9, 9)
        digits = np.arange(1, 10)
        assert np.all(np.sort(grids, axis=2) == digits)
        assert np.all(np.sort(grids.transpose(0, 2, 1), axis=2) == digits)
        boxes = grids.reshape(-1, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4).reshape(-1, 9, 9)
        assert np.all(np.sort(boxes, axis=2) == digits)
        compact = np.ascontiguousarray(solutions, dtype=np.uint8)
        keys = compact.view(np.dtype((np.void, 81))).reshape(-1)
        assert len(np.unique(keys)) == len(keys), "Full grids overlap or repeat"
        with Pool(args.workers) as pool:
            for valid in tqdm(
                pool.imap(unique, puzzles, chunksize=128),
                total=len(puzzles),
                desc="Verify unique solutions",
            ):
                if not valid:
                    raise ValueError("Source puzzle is not uniquely solvable")
        mask = np.tile(np.array([0] * 90 + [1] * 90, dtype=np.int64), (205000, 1))
        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {"input_ids": tokens[:200000], "attention_mask": mask[:200000]}
                ),
                "validation": Dataset.from_dict(
                    {"input_ids": tokens[200000:], "attention_mask": mask[200000:]}
                ),
            }
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(destination))
        manifest = {
            "source": str(args.source.resolve()),
            "protocol": "SDM J.4",
            "train": 200000,
            "validation": 5000,
            "clues": 30,
            "unique_solutions_verified": True,
            "disjoint_full_grids": True,
            "vocabulary": 14,
            "prompt_tokens": 90,
            "token_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
            "source_files": {
                str(path.relative_to(args.source)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in args.source.rglob("*.arrow")
            },
        }
        path = destination / "paper_manifest.json"
        path.write_text(json.dumps(manifest, indent=2))
        artifact = wandb.Artifact("sudoku-paper-data-manifest", type="dataset")
        artifact.add_file(str(path))
        run.log_artifact(artifact)
        run.summary.update(manifest)
        print(destination)


if __name__ == "__main__":
    main()

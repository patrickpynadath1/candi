---
name: sddc-archive
description: Pack finished SDDC run trees into a portable, checksummed archive without checkpoints (keeping everything analyze.py needs, mtimes and provenance), verify the archive reproduces the analysis, and restore it elsewhere. Use when runs finish, before cleaning up a server, when moving results between machines, or when publishing results.
---

# Archive runs

Run trees hold checkpoints (often most of their size) next to small result files. An archive
keeps the results and drops the bulk, so results outlive the server and can be re-analyzed
anywhere.

## 1. Look before packing

```bash
uv run --locked python -m core.analysis.status $SDDC_STORAGE_ROOT/runs/<name>
uv run --locked python scripts/archive_runs.py $SDDC_STORAGE_ROOT/runs/<name> \
  --output $SDDC_STORAGE_ROOT/archives/<name>.tar.gz --dry-run
```

`--dry-run` lists how many files (and MiB) are kept and dropped, by reason. Archiving
unfinished cases is allowed; `index.json` records each case's state.

## 2. Pack and verify

```bash
uv run --locked --extra tuning --extra scoring python scripts/archive_runs.py \
  $SDDC_STORAGE_ROOT/runs/<name> [more roots] \
  --output $SDDC_STORAGE_ROOT/archives/<name>-$(date +%Y%m%d).tar.gz \
  --verify experiments/<example>/analyze.py
```

- **Dropped**: `*.ckpt`, `*.pt`, `*.safetensors`, `*.bin`, locks, temporary files, W&B and
  Ray directories, and files over `--max-file-mb` (default 200).
- **Kept**: everything else, with original mtimes (training hours are derived from them):
  `protocol.json`, configs, Optuna databases, every candidate's results, final results and
  generated records.
- **Added**: `index.json` and `summary.csv` (case, state, task, recipe, selected policy, test
  score), `provenance/` (git revision, uncommitted diff, `uv.lock`, `pyproject.toml`, package
  versions, the command), and `SHA256SUMS`.
- `--verify` extracts the archive next to it, runs the analysis on the original roots and on
  the archive, and fails unless both write the same results.

An archive refuses to overwrite an existing file (`--force` to replace). It may contain
absolute paths and user names from the run machine; check before sharing it publicly.

## 3. Restore

```bash
mkdir -p restored && tar -xzf <name>.tar.gz -C restored
(cd restored && sha256sum -c SHA256SUMS)       # shasum -a 256 -c on macOS
uv run --locked --extra tuning --extra scoring python experiments/<example>/analyze.py \
  restored/runs/<root name> --output analysis/<name>
```

Analyses that need checkpoints (the continuous-schedules schedule panels) skip those parts
from an archive and say so; pass `--no-schedules` to skip them explicitly.

## Done when

The archive exists, `--verify` printed `verified`, and only then may the user decide whether
to free the original run tree. Deleting runs or checkpoints needs their explicit go-ahead.

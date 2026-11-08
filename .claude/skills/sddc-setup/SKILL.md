---
name: sddc-setup
description: Install SDDC, prepare its datasets and confirm the checkout works (environment, data caches, tokenizers, a 2-step CPU train/evaluate, quick tests). Use on a fresh clone, a new machine or server, after changing uv.lock, or when runs fail for environment reasons.
---

# Set up SDDC and confirm it works

## 1. Pick the install for this machine

The lock (`uv.lock`) pins Python 3.10 and PyTorch 2.5.1 built for CUDA 12.1, with wheels
for Linux x86_64 and Windows only.

| Machine | Install | What runs |
| --- | --- | --- |
| Linux x86_64 with an NVIDIA GPU | `uv sync --locked --extra tuning --extra scoring --extra tracking --extra mazes` | everything |
| Linux x86_64, CPU only | the same command (the CUDA wheel runs on CPU) | the full test suite, smoke runs, tiny cases; this is what CI does |
| macOS or another platform | `uv sync --locked` fails: the lock has no torch wheel here | torch-free tests via `uv run --no-project --python 3.10 --with pytest python -m pytest core/tests/test_remote_submission.py`; GPU work goes to a server (skill `sddc-remote`) |

Extras: `tuning` (Optuna, needed for sweeps), `scoring` (sacrebleu, BLEU on TinyStories),
`tracking` (W&B, optional), `mazes` (maze tasks). Always run through the lock:
`uv run --locked ...`, never system Python or `pip install`.

On AMD GPUs the lock does not apply; install a ROCm torch build into a separate environment
and say so in any results (the blog's MI210 runs used torch 2.9.1+rocm6.3).

## 2. Choose the storage root

Set `SDDC_STORAGE_ROOT` to a large persistent volume; the default is `~/.cache/sddc`. Plans,
runs, data caches and archives all go under it. On a shared server, never use `/tmp` or a
nearly full root disk.

## 3. Prepare data (only the tasks you need)

- **Sudoku, N-Queens, mazes**: generated on first use; nothing to do.
- **text8**: `uv run --locked python scripts/prepare_data.py text8` (checked by SHA-256),
  into `$SDDC_STORAGE_ROOT/data_cache/text8/raw`.
- **TinyStories**: download the parquet files (`huggingface-cli download roneneldan/TinyStories`,
  see `docs/tasks/tinystories.md`), set `TINYSTORIES_DATA_DIR` to them, then
  `uv run --locked python scripts/prepare_data.py tinystories --parquet $TINYSTORIES_DATA_DIR`
  to build the published cohort (`allowed_ids.json`). The BPE tokenizers are committed in
  `artifacts/tokenizers/tinystories-bpe/`.

The task suites (`configs/task/battery/`) set `download: false`, so a missing cache is an
error, not a silent download.

## 4. Check

```bash
uv run --locked --extra tuning --extra scoring python scripts/doctor.py --smoke --tests
```

`scripts/doctor.py` prints one line per check (Python, platform, uv, torch and GPU, each
extra, storage root, data caches, tokenizer checksums, config composition), then trains the
2-step CPU smoke model (`+experiment=smoke`) under `$SDDC_STORAGE_ROOT/doctor/`, evaluates
it, and runs a quick test subset. It exits 1 on any `fail`; a `warn` is something optional
(an extra, a dataset you may not need).

Before trusting a change to the environment, also run the full suite once:
`uv run --locked --extra tuning --extra scoring --extra tracking --extra mazes pytest -q core/tests`
(about 15 minutes on CPU). GPU-only tests skip on CPU. The byte-level sampler snapshots run
only with `SDDC_BYTE_SNAPSHOTS=1` on the machine that recorded them.

## Done when

`doctor.py --smoke --tests` reports `0 fail`, and every task you plan to use has its data
check at `ok`.

## Pitfalls

- `uv run` without `--locked` may re-resolve and change the environment.
- A dataset prepared by hand rather than by `scripts/prepare_data.py` will not match the
  published splits.
- The default `main.py` with no overrides is a full GPU training run, not a smoke test.

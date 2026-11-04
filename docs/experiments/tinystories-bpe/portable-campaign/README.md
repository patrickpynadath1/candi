# Portable campaign handoff (draft)

This is a new campaign revision. `inventory.json` records the requested method
matrix; it is **not** an executable nested-tuning manifest. Do not launch it.
Task selection is fixed: Sudoku Hard, TinyStories 10k BPE, Text8 and the
TinyStories vocabulary ladder (115 cases per seed). Dataset partitions must
be finalized before export. Slurm/Ray setup and smoke execution are implemented
in [the campaign scripts](../../../../scripts/campaign/README.md).

The original 49-case TinyStories campaign was launched on Gautschi on November 2.
Its frozen cohort, exported configs and partitions are on scratch, currently
unavailable during the November 3 outage. Do not substitute fresh partitions
and label them as the same experiment. New experiments may use new, explicitly
versioned partitions; old checkpoints require an exact protocol comparison.

## Requested revision

- The original TinyStories method matrix: 32 diffusion/context cases and one AR
  baseline at vocabulary 10,000. Its vocabulary ladder adds 16 unique cases.
- Repeat all 33 methods for generated Sudoku Hard and Text8. Sudoku-Extreme
  and mazes are excluded.
- Enable `model.adaLN` and `algorithm.time_conditioning` in all 12 block/context
  cases. Clean-context E2D2 encoders remain unconditioned.
- Add the existing `ForwardResamplingSampler` to inference tuning after checking
  each algorithm/task combination. Retain the same sequence-level NFE budget;
  calculate block budgets with `core.sweeps.budget.sampler_steps`.
- Use new case identities/output paths for expanded tuning policies. Do not
  mutate pinned protocols or silently reuse old sampler selection results.
- Deliver a compressed results archive, not a paper.

## Environment and execution

Pin a Git commit and use the same locked uv environment on the driver and every
worker. Install with `uv sync --locked --extra tuning --extra scoring --extra
tracking`; include `--extra mazes` when using mazes. Record Python, PyTorch, CUDA,
Ray, GPU model and driver versions. The campaign source must include the local
AR validator fix allowing `algorithm.predictor.commitment` in inference policies.

Use Hydra multirun preparation through `core.sweeps.prepare`, followed by its
`collect` command, and execute with `core.sweeps.nested_tuning`. No new scheduler
is needed. Remove historical reference manifests and machine-specific labels.
Each training/evaluation task requests one GPU. Set case concurrency according
to available GPUs, not the number of nodes; keep inference concurrency at one
per case initially. Multi-GPU nodes can host independent cases.

A multi-node deployment needs a reachable Ray head, matching worker environments,
and shared paths for source/configs, datasets, tokenizers and checkpoints. The
driver's Optuna databases and lock files need reliable filesystem semantics.
Do not run a global `ray stop` on shared machines. Configure authentication outside
the manifest and never include credentials in the delivery bundle.

## Data and tokenizer prerequisites

Copy existing `bpe-260.json`, `bpe-500.json`, `bpe-1000.json`, `bpe-5000.json` and
`bpe-10000.json` from the original tokenizer artifacts. Do not retrain them.
Record and verify SHA-256 hashes. The old audit manifest contains source hashes;
do not assume it contains tokenizer-file hashes without checking.

Download TinyStories parquet shards at revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. Recover and transfer the frozen cohort
allowlist and tuning/test stable IDs. Map stable IDs to prepared indices for
each tokenizer and assert identical retained cohorts. Prepare caches once before
starting concurrent workers. Do not put datasets or tokenizer caches in Git.

Default shared training budget is 20,000 updates, global batch 256, hidden width
512, eight heads, LR 3e-4 and 2,500 warmup updates. Microbatch/accumulation may be
adapted to hardware while preserving the global batch. Validate task-specific
lengths, output vocabulary, masking, block size and primary metrics separately.
Never use test results for sampler selection.

## Required delivery archive

Deliver a versioned `.tar.gz` plus a SHA-256 checksum and machine-readable index.
The archive should include:

1. Exact source commit, patch if applicable, lockfile and launch/environment record.
2. Final manifest, pinned configs/protocols, data/tokenizer hashes and split IDs.
3. Per-case final checkpoint and evaluation config, including learned time maps.
4. Optuna studies, per-policy selections, final selection and final metrics JSONs.
5. Generated answers and per-example evaluation records for tuning/final runs.
6. Summary CSV/JSON with task, method, seed, parameters, training time, GPU memory,
   selected sampler, measured NFE, latency and task metrics.
7. Explicit completed/failed/incomplete status for every requested case.

Checkpoint payloads may be split into a separate archive with their own checksum
index. Exclude credentials, raw datasets, environments, runtime caches and Ray
session directories. Archive only after writers finish; a live SQLite database
must be backed up using SQLite's backup API rather than copied mid-transaction.

## Launch gates

Before calling the handoff ready: export and validate the exact case count,
check all paths on workers, verify data/partition/tokenizer hashes, construct
every model, test every sampler family at the actual NFE budget, run short
train/checkpoint/evaluate smoke checks, and verify the delivery archive can be
read and its checksums pass. These checks have not yet been performed for this
expanded campaign.

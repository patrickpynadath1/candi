# Reasoning task ablations

The structured-task pipeline supports Sudoku-Extreme, Sudoku-Shah, Zebra 4×4,
Zebra 5×5, mixed Zebra (3–6 attributes/houses), and EqR Maze-Unique 30×30.
The shared matrix remains **eight pure-continuous and eight hybrid models** in
`configs/sweeps/sudoku_rank.yaml`; task settings live in
`configs/sweeps/reasoning_tasks.yaml`. Six profiles × 16 models × three seeds
produce 288 training jobs. Shah→Extreme adds evaluations of Shah checkpoints,
not training jobs. The generated easy/medium/hard Sudoku runner is unchanged.

## Run commands

Use the project virtual environment. Full `prepare`, `train`, and `eval` actions
require `--execute`; without it they print the plan. `smoke` executes short CPU
runs. These commands do not provision cloud resources.

```bash
.venv/bin/python new_scripts/download_reasoning.py
.venv/bin/python new_scripts/reasoning_ablation.py plan
.venv/bin/python new_scripts/reasoning_ablation.py prepare --execute
.venv/bin/python new_scripts/reasoning_ablation.py train --profiles sudoku_extreme zebra_4x4 maze_unique --gpus 0 --execute
.venv/bin/python new_scripts/reasoning_ablation.py eval --profiles sudoku_extreme zebra_4x4 maze_unique --gpus 0 --execute
.venv/bin/python new_scripts/reasoning_ablation.py aggregate
```

`--profiles`, `--variants`, `--modes`, `--seeds`, `--prepared`, and `--root` select
subsets and locations. Training and evaluation resume completed jobs and record
commands, logs, status, configuration, and dataset fingerprints. Evaluation uses
the final fixed-budget `last.ckpt`, and checks checkpoint/data hashes before
reusing results. NFE budgets default to 16/32/64/128/256. Results include per-seed
and mean/std CSVs, exact success, task metrics, actual NFE, and elapsed time.
The defaults are initial comparison budgets, not reproductions of source-paper
training compute or tuned hyperparameters.

For a local smoke test, use a separate prepared cache and output root:

```bash
.venv/bin/python new_scripts/reasoning_ablation.py prepare --prepared /tmp/reasoning_smoke_data --limit 256 --execute
.venv/bin/python new_scripts/reasoning_ablation.py smoke --prepared /tmp/reasoning_smoke_data --root /tmp/reasoning_smoke_runs
```

Smoke caches are labeled subsets and are rejected for full sweeps. Full conversion
is a separate operation from downloading and can take substantial CPU time for
million-record releases. Conversion writes memory-mapped Arrow; training workers
do not repeatedly load the raw Zebra pickle. Partial conversions remain in a
`.preparing` directory for inspection; retry with a fresh destination.

## Intermediate generation evaluations

Both sweep runners now enable `ReasoningEvaluationCallback` automatically:
**every 1,000 optimizer steps, 128 fixed puzzles, 45 sampling steps**, seed 42,
with evaluation batches of four. The subset and sampling seed stay fixed across
checkpoints and model seeds. Smoke runs use four puzzles/four sampling steps at
each optimizer step instead.

Sudoku hard and Zebra monitor their validation splits. The full-training Extreme
and Maze profiles monitor fixed official-test subsets while retaining all 1,000
training groups. These test curves are monitoring only: the callback does not
select checkpoints, stop training, or tune schedules. Final reporting still uses
the declared final checkpoint and evaluation protocol.

Each run writes `monitoring/subset.json` (source IDs and protocol) and
`monitoring/step_XXXXXXXX.json` (metrics and predictions). Scalars are also sent
to the configured training logger under `monitor/<task>/<split>/...`. Zebra logs
exact assignment, constraint satisfaction, malformed output and per-layout
results; Maze logs exact match, valid/optimal path rates and length error.

The callback evaluates EMA weights when enabled, restores training weights and
random states, and retains its last evaluated optimizer step in checkpoints.
It runs independently of Lightning validation, including when validation is off.
The implementation passed 101 focused regression tests, eight CPU train/eval
smokes across the four task families, and a BF16 GPU Maze train/resume check
with EMA and validation disabled.

For direct `main.py` runs, add `+callbacks=reasoning_eval`. Defaults are in
`configs/callbacks/reasoning_eval.yaml`; overrides include
`callbacks.reasoning_eval.every_n_train_steps=1000` and
`callbacks.reasoning_eval.num_steps=45`.

## Training and split protocols

**Extreme and Maze primary profiles use all 1,000 released training groups**, a
fixed step budget, and no validation-based checkpoint selection. Extreme has
1,001,000 rows because each base puzzle includes the original and 1,000 released
augmentations. Its test split has 422,786 puzzles. Maze has 1,000 train and 1,000
test mazes. These are the downloaded EqR artifacts, not newly sampled substitutes.

Development experiments can instead hold out training groups. For example:

```bash
.venv/bin/python new_scripts/prepare_reasoning.py --task sudoku_extreme --output data_cache/reasoning_dev/sudoku_extreme
.venv/bin/python new_scripts/reasoning_ablation.py train --profiles sudoku_extreme --prepared data_cache/reasoning_dev --root outputs/reasoning_dev --gpus 0 --execute
```

The default development probability is 5%, determined by a seeded group hash;
the manifest records exact counts. This development profile trains on fewer than
1,000 base puzzles and must be reported as such. Augmentation siblings stay in
the same split. To prepare an all-training artifact directly, add
`--full-training`; the runner disables validation and monitored checkpointing.

Shah and Zebra derive development splits from released training puzzles, with
canonical puzzle/clue groups kept together. Official test data never enters
training or model selection. Shah training has 1,804,463 records and test has
100,000. Zebra training has 1,499,933 records and test has 100,000; 4×4 has
93,604/6,393 and 5×5 has 93,767/6,229 before development splitting.

Shah→Extreme retains the same Sudoku tokenizer, 180-position layout, model,
checkpoint, and inference budget. Preparation also produces an exact-puzzle
cross-source overlap audit, which the OOD evaluator checks against source hashes.
The downloaded release has zero exact overlaps between all Shah training puzzles
and Extreme test. This does not establish non-equivalence under Sudoku symmetries.

Manifests record source URLs/revisions where available, SHA-256 file hashes,
preprocessing version, vocabulary/layout, seeds, split/group counts, gold checks,
and exact puzzle overlap. `selected_ids.jsonl` records selected source/group IDs.
The released Zebra files contain train/test duplicates. Preparation preserves the
official test set and removes every matching canonical clue group from training
and validation, including clue-order permutations. Manifests record the excluded
record/group counts and verify zero prepared train/test puzzle overlap.

## Representations and scoring

- **Sudoku:** existing 180-token puzzle/solution representation. Shah solution and
  strategy tuples are decoded by coordinates; only given cells enter the prompt.
  Metrics include exact solution, grid validity, clue consistency, and digit accuracy.
- **Maze:** 902 prompt positions and 900 output positions (1,802 total). Preserve
  EqR walls, spaces, start, goal, and internal-path labels. An independent BFS
  checker tests connectivity, wall avoidance, no extra path components, shortest
  length, and uniqueness. Exact and optimal-path success matter more than
  background-dominated cell accuracy.
- **Zebra:** variable-length clue prefix followed immediately by up to 36 canonical
  assignment targets, then trailing padding to 512 positions. The configured
  476-token prompt length is a maximum, not a fixed answer offset. Vocabulary has
  24 symbols and is fixed before training.
  Source sequences are cut at `ANSWER`; solution triples and solver ordering
  never enter the prompt. Targets are attribute-by-house tables. Report exact
  assignment, constraint satisfaction, malformed output, and per-layout results.
  Released `inbetween` triples are serialized left-to-right: first < second < third.
  This convention was checked against all 100,000 test labels and small solver
  audits. The paper describes the seven clue types in
  [Appendix A](https://arxiv.org/html/2409.10502v1#A1).

The task `attention_mask` is the loss/corruption mask. Prompt and trailing padding
tokens stay fixed and are excluded from CE/adaptive statistics. It is not a transformer
padding-attention mask: trailing padding remains visible context. No generation-time
repair, solver guidance, or task-specific output filtering is applied. Rank
calibration uses the actual model embedding dimension and tokenizer vocabulary,
including input-only symbols; it retains the existing iid approximation.

## Validation and pilots

Validation evidence is recorded in `analysis_results/reasoning_task_validation.json`.
The full regression suite passed 606 tests; the final focused task suite passed
88 tests, including two subsequently added runner safeguards.
The implementation was exercised with 96 two-step CPU train/checkpoint/eval runs
across all six profiles and all 16 models, including 16 additional Shah→Extreme
evaluations. These are integration checks, not benchmark accuracy results.

All released Sudoku labels and all 2,000 Maze labels passed independent checks.
All 100,000 Zebra test labels passed the constraint checker. The full Zebra
training archive was inventoried; full preparation checks every selected label.

GPU pilots ran on an RTX A6000. For the two representative hybrid adaptive
variants, configured ~28.6M-parameter models used approximately 2.4–3.9 GiB peak
allocated memory at batches 64 (Sudoku), 16 (Zebra), and 4 (Maze). Measurements
used float32 without EMA and ten timed optimization steps; they are preliminary
capacity checks, not final training-throughput estimates. Two-step BF16
train/checkpoint/evaluation checks also exercised both variants on the full
1,000-example Maze training profile. The sweep trainer uses
its configured precision/EMA, so rerun a pilot when those settings change.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python new_scripts/reasoning_pilot.py --prepared /tmp/reasoning_smoke_data --output /tmp/reasoning_gpu_pilot --device cuda
CUDA_VISIBLE_DEVICES=0 .venv/bin/python new_scripts/reasoning_pilot.py --prepared /tmp/reasoning_smoke_data --output /tmp/reasoning_capacity --device cuda --production --steps 10
```

The 200-step, two-example learning probes reduced loss on all four task families
for both tested variants and preserved conditioning. They did not establish
near-perfect tiny-set generation or task-level convergence. Full benchmark
training, hyperparameter selection, and convergence comparisons remain to run.
A GPU hidden by the outer execution sandbox is accessible with an approved
unsandboxed command; the server has eight A6000s. Do not interpret sandbox CUDA
visibility as server hardware availability.

Zebra size OOD, Maze length OOD, a separate conditioning encoder, and alternative
reasoning-trace supervision remain additional ablations, not part of this matrix.

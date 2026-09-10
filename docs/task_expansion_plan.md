# Plan: extend the rank-schedule ablations to structured reasoning benchmarks

Status: implemented task adapters, scoring, preparation, and sweep integration.
See [reasoning_tasks.md](reasoning_tasks.md) for commands, validation evidence,
protocol distinctions, and remaining benchmark experiments.

## Source findings and protocol choices

- **Sudoku-Extreme:** use the released HRM/EqR small-sample artifact or reproduce
  its builder. HRM's `num_aug=1000` includes the original as well as the augmented
  instances: 1,000 base puzzles produce 1,001,000 training records. Preserve base
  puzzle IDs and augmentation groups. The builder stores digits with a +1 offset;
  convert explicitly into our current Sudoku tokenizer.
  [Builder](https://github.com/sapientinc/HRM/blob/main/dataset/build_sudoku_dataset.py).
- **Sudoku-Shah and Zebra:** the current repository README lists approximately
  1.8M/0.1M Sudoku train/test and 1.5M/0.1M Zebra train/test records. Sudoku records
  contain givens and solution/strategy tuples; Zebra records contain symbolic
  clues, the assignment table, and solver ordering. The earlier paper describes
  around 320k Zebra puzzles. Treat release identity and actual counts as metadata;
  do not infer a release history from the discrepancy.
  [Repository](https://github.com/kulinshah98/llm-reasoning-logic-puzzles),
  [paper v1](https://arxiv.org/html/2409.10502v1).
- **Maze-Unique:** use the EqR artifact, not HRM Maze-1k. EqR specifies 30x30
  perfect mazes, unique shortest paths, and 1,000 examples in each train/test
  split. Its repository supplies downloads through `locuslab/EqR-data` and a
  builder. Prefer the released split for direct comparability.
  [EqR data instructions](https://github.com/locuslab/EqR#data),
  [dataset definitions](https://arxiv.org/html/2605.21488v1#A3).
- **FRM:** the public repository currently says implementation coming soon.
  Use its paper as an experimental-protocol reference, not an available adapter.
  Its v1 discusses Shah-trained, evaluation-only Extreme and distinct Zebra
  layouts; these should be separate experiment profiles.
  [Repository](https://github.com/helblazer811/Flow-Reasoning-Models),
  [paper v1](https://arxiv.org/html/2606.29150v1).

## Milestone 1: shared task and dataset contract

Extend `tasks/registry.py` to register task loaders, tokenizers, encoders, and
scorers. Refactor `tasks/evaluation.py` to obtain examples, prompt masks, answer
positions, metadata, and metrics from the task rather than hardcoding
`get_sudoku_dataset`, a shared `prompt_len`, and the Sudoku scorer. Keep existing
Sudoku results and command-line entry points backward-compatible.

The canonical example should retain:

- stable source/base-puzzle ID, split, source metadata and augmentation identity;
- typed puzzle/clues and canonical solution;
- token IDs, conditioning mask, target/loss mask and padding mask;
- answer layout and task-specific evaluation metadata.

The existing `attention_mask` is largely used as a loss/corruption mask. Do not
silently reuse it as a transformer padding-attention mask. Start with fixed
layouts or length buckets; make padding uncorrupted, excluded from CE and
adaptive statistics, and consistently treated in training and sampling. If true
attention masking is introduced, propagate it through all relevant backbones.

Write immutable dataset manifests with upstream commit/revision, download URL,
file hashes, actual split counts, preprocessing version, tokenizer vocabulary,
seeds, selected example IDs, augmentation settings and benchmark profile.
Store converted data in Arrow or memory-mapped arrays; avoid eager replication
of million-example releases. Derive development subsets from training groups;
keep official test splits isolated. Verify the selected small-sample artifact
rather than assuming that a newly sampled set matches a paper's 1k subset.

Deliverables: shared task contract, reusable cache/provenance utilities, and
unchanged existing Sudoku regression results.

## Milestone 2: Sudoku-Extreme and Sudoku-Shah

Add distinct data IDs/configurations: `sudoku_extreme`, `sudoku_shah`.

Reuse our 180-token puzzle/solution representation and Sudoku checker. Keep
source-native difficulty ratings and clue counts; do not relabel Extreme with
our generated easy/medium/hard categories.

For Extreme, support a pinned released 1k augmented profile and an optional full
profile. Load official augmentations first for reproducibility; offer seeded
on-the-fly Sudoku-preserving transforms later as a separately recorded profile.
Any development split must happen by base puzzle before augmentation. Never
place siblings of an augmented puzzle in train and development splits.

For Shah, reconstruct givens using the given-cell marker and reconstruct the
solution by coordinates. Do not put solution values, strategy labels, or solver
ordering into the conditioning input. Initially train on puzzle/final-solution
pairs; intermediate reasoning supervision would be another ablation.

Keep two Extreme evaluations distinct:

1. Train on Extreme-1k and evaluate the official Extreme test split.
2. Train on Sudoku-Shah and evaluate Extreme as out-of-distribution data.
   This generalization experiment is explicitly included in the requested plan.

The OOD profile must not train or choose hyperparameters/checkpoints on Extreme.
Cross-source duplicate checks should accompany any OOD claim. Report exact
solution rate, clue consistency, valid completed grid rate and digit accuracy.

Deliverables: loaders, provenance manifests, source-specific configs, valid
round trips and independent Sudoku-solver checks.

## Milestone 3: Maze-Unique

Add `tasks/maze.py` and `configs/data/maze_unique.yaml`. Import EqR arrays using
the artifact's symbol metadata; verify padding and start/goal/path conventions
before mapping them into our tokenizer.

Initially serialize a fixed 900-cell input board and fixed 900-cell output
board/path labeling with delimiters. Keep input maze and endpoints clean. This
uses roughly 1,802 positions under concatenation, versus current Sudoku's 180;
profile memory and batch size before scaling. A separate conditioning encoder
could reduce cost, but would be an architecture ablation, not a prerequisite.

Implement an independent BFS-based checker. Validate source labels, endpoint
inclusion, wall avoidance, connectivity, no spurious disconnected path cells,
path length and shortest-path uniqueness. Score exact target match, valid path,
optimal path, and length error. Do not use background-dominated cell accuracy
as the headline metric. Preserve EqR's endpoint/path-label convention in exact
match scoring.

Deliverables: fixed-layout tokenizer/loader, path checker, source-label checks,
and CPU train/resume/eval smoke for every configured model variant.

## Milestone 4: Zebra

Add `tasks/zebra.py`, a deterministic symbolic tokenizer, and
`configs/data/zebra.yaml`. Parse clues into typed relations; validate every clue
type against the upstream semantics. Retain house and attribute counts.

The input contains clues and dimensions only. The target is a canonical
attribute-by-house assignment table; no solution-dependent ordering or solver
trace is supplied. Begin with a pinned 4x4 profile, then enable 5x5 and mixed-size
3-to-6 entity/attribute profiles with explicit padding and bucketing. Do not
silently truncate long clue lists. Fix vocabulary and answer layout across
splits rather than fitting them using test labels.

Implement a standalone constraint checker covering all relation types and
per-attribute bijections. Score exact assignment, all-constraints-satisfied,
cell accuracy, malformed output rate and per-layout solve rate. Use a small
constraint solver to cross-check parsing and unique solutions in fixtures and
source audits. Keep any generation-time repair or constraint-guided selection
out of the primary baseline.

Deliverables: deterministic parser/serializer, independently tested checker,
layout-specific profiles, and all-variant training/evaluation smoke runs.

## Milestone 5: generalized sweeps and acceptance gates

Extract the current Sudoku runner into a generic reasoning runner while
retaining the old wrapper. Keep the 16-model matrix independent of task profiles:

    task/profile x allowed variant/mode x seeds x evaluation budgets

Profiles specify dataset provenance, sequence layout, vocabulary, training
budget, evaluation splits and task metrics. A profile need not have an
`easy/medium/hard` field. Keep the current eight pure/eight hybrid restrictions.
At three seeds, each selected training profile contributes 48 training runs;
OOD evaluations reuse checkpoints rather than add training runs.

Use actual embedding dimension for spherical rank calibration and actual
candidate vocabulary for vocabulary-dependent truncation. Tokenizers may have
input-only clue symbols; document the answer alphabet and any output filtering,
and keep those choices identical across compared algorithms. Do not silently
assume Sudoku's vocabulary size or add task-specific output restrictions to just
one method. Calibration remains an iid model; retain empirical diagnostics where
available.

Acceptance gates, in order:

1. Source manifest and split/group integrity checks; verified token round trips.
2. Gold labels score perfectly; deliberately corrupted outputs fail correctly.
3. All 16 configurations produce finite loss/gradients, preserve conditioning,
   and restore adaptive schedules/checkpoints on small fixtures of every task.
4. Short tiny-set overfit experiments show the representation can be learned.
5. A GPU pilot establishes memory, throughput and suitable batch sizes.
6. Full sweeps use fixed splits, comparable model capacity and declared training
   budgets; report actual NFE, wall time, FLOPs estimates where available, seed
   variation and task-specific exact success. Multiple candidates, verifier
   calls, restarts or solution repair must be separate, fully charged protocols.

Source archives have now been downloaded and counted; source-label checks and
small constraint-solver audits are complete as described in the implementation
guide. Full model convergence and benchmark sweeps remain experimental work.


## Confirmed evaluation scope

Include both in-distribution Sudoku benchmarks and the Shah-to-Extreme OOD
experiment. Shah-to-Extreme reuses Shah-trained checkpoints, with model selection
on Shah development data only. Fix the Extreme evaluation subset or full split
before running comparisons and record cross-source overlap checks.

Zebra's initial benchmark uses the released train/test partitions, reports
results separately by (number of houses, number of attributes), and includes
pinned 4x4 and 5x5 profiles plus a mixed-size profile once variable layouts work.
Size generalization is a proposed additional protocol, not an automatic property
of the mixed-size benchmark: e.g. train on layouts with both dimensions <=5,
then evaluate on held-out layouts containing a dimension of 6. Reserve the
vocabulary and input capacity in advance; explicitly distinguish new sizes from
unseen symbol IDs. Select checkpoints only on in-distribution development data.
No size-generalization result should be called a released-paper reproduction
without matching that paper's actual split and encoding.

Maze's initial benchmark is the released Maze-Unique 30x30, 1k/1k train/test
split, using unique-shortest-path correctness and exact match. A proposed extra
generalization protocol holds grid size fixed and tests longer shortest paths
than the training range. Use a separately versioned generated test set, enforce
uniqueness, and deduplicate underlying mazes across splits, even when endpoints
differ. This is a custom OOD experiment rather than the released benchmark.
Larger-grid transfer is deferred: it also requires decisions about sequence
length and positional encoding and would confound the first schedule comparison.

Only the Shah-to-Extreme OOD experiment is currently explicitly requested;
Zebra size transfer and Maze path-length transfer are proposed extensions.

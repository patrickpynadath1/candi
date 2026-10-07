# Single-cell gene-expression tasks and matched sweep

The attached draft *Single-cell Gene Expression Generation with Hybrid
Discrete-Continuous Diffusion* (§5.1) evaluates the following datasets:

| Task/data name | Cells reported | Genes reported | Cell types | CFGen label column |
|---|---:|---:|---:|---|
| `pbmc3k` | 2,638 | 8,573 | 8 | `cell_type` |
| `dentategyrus` | 18,213 | 17,002 | 14 | `clusters` |
| `pbmc_covid` | 422,220 | 2,000 | 29 | `cell_type` |

Sources: [CFGen code](https://github.com/theislab/CFGen),
[CFGen dataset archive](https://zenodo.org/records/15031364), and
[DCM code](https://github.com/sanjukta7/aivc-dcm). CFGen's dataset configs specify
`layers['X_counts']` and the label columns above. The archive's verified members
are `datasets/NAME/NAME_train.h5ad` and `datasets/NAME/NAME_test.h5ad`.
The 15.1 GB archive's published MD5 is `c0a6582b05e6ce758811e58d703c0aff`.
The preparation command extracts only the six relevant files, not other datasets.

The draft's Table 3 calls PBMC3K's labels “8 perturbations”; §5.1 and CFGen identify
these as **cell types**. These tasks do not manufacture perturbation labels.
The published CFGen paper reports a different PBMC COVID cell count (344,820).
Actual prepared dimensions and split counts are recorded rather than enforced
from either paper. The draft does not fully specify binning, splits, or metric
normalization, so this is a reproducible new benchmark, not a claim to reproduce
its numerical tables.

## Prepare

```bash
uv sync --extra single-cell
uv run --extra single-cell python examples/prepare_single_cell.py \
  --source-dir /absolute/path/to/datasets
```

Alternatively pass `--archive /absolute/path/to/datasets.zip`, or `--download`
to download the verified CFGen archive. Downloading/preparation are explicit;
previewing the sweep performs neither. Install `--extra tracking` as well if
using W&B. The existing AnnData splits are preserved: 10% of the upstream
training split becomes validation (seed 42), and the upstream test set is kept
separate. Missing labels, overlapping cell IDs, fractional/negative counts,
and mismatched gene sets are rejected. Test genes are aligned to training order.
There is no silent fallback to normalized `X`; use `--layer X` explicitly if
that is the intended integer-count source.

The default vocabulary retains all training counts exactly, plus a tail token
at `max_training_count + 1` for unseen larger counts. Zero remains zero.
`--count-cap 100` explicitly collapses counts of 100 or more to 100 and reduces
model/state memory. This is optional and is **not a binning rule specified by the
paper**. Normalized/log data are never discretized by truncating floats. All
choices, gene ordering, label mapping, source paths/sizes, split indices, and tail
fractions are recorded in the prepared directory. Vocabulary fitting uses only
the internal training split. Memory-mapped int32 arrays allow worker processes
to share prepared data without loading the full dense dataset into each worker.
Preparation reads one dataset's two AnnData files into memory, but densifies only
128 rows at a time; plan for RAM for their sparse arrays and disk for dense caches.

## Task semantics

`task=single_cell data=pbmc3k` (or the other dataset names) uses the normal Task
interface for training, callbacks, and evaluation. Set `data.vocab_size` from
`metadata.json`; the sweep does this automatically. Set `task.length` to the
number of genes, plus one for conditional generation. Prepared metadata is
validated against the configuration before training.

Each gene is one fixed sequence position and each token is a count. Unconditional
prompts contain no observed genes. Conditional prompts prepend a fixed cell-type
label; label IDs share the count vocabulary but occupy a distinct position.
This prefix is excluded from corruption, loss, and metrics. Labels are not
predicted and reference expression is never included in the generation prompt.
This implements conditioning through the existing backbone/task contract, not
the draft's unspecified conditioning architecture.

The primary evaluation split is `test`; training loss validation uses only the
internal `validation` split. Primary metric is squared biased RBF MMD, minimized.
Metrics include empirical 2-Wasserstein distance, zero fraction, and mean count.
Both distributions are library-normalized to 10,000 then log1p-transformed for
MMD/W2; sparsity and mean counts are measured before normalization. All-zero
cells remain zero. Squared distances are divided by gene count. MMD bandwidth
is the median positive squared distance between reference cells (1 if degenerate).
W2 uses exact equal-weight Hungarian matching and returns the square root of the
mean matched squared distance. Conditional runs also report per-cell-type
metrics and their macro means; singleton groups are retained and report their
sample counts. Invalid generated tokens cause an error instead of being clipped.
Scores are measured in the prepared count representation, including any tail bin;
there is no recovery of original counts above the cap.

Generated and reference counts are retained in result records with cell labels,
indices, and measured NFE. Full gene vectors and exact OT can make evaluations
large; reduce `--max-examples` for pilots. These explicit metric conventions are
not asserted to equal the draft's unspecified implementation.

## Reproduce the Sudoku sweep structure

The source is the **full-hint** comparison in
[`results/sudoku-hard-5k/full-hints/report.md`](../../results/sudoku-hard-5k/full-hints/report.md),
using `adaptive_hybrid_feedback_sweep.py`'s continuous/hybrid models and the
discrete feedback sweep. Sudoku hint removal has no direct single-cell equivalent.

```bash
# Preview (metadata can be absent; vocabulary is then unresolved).
uv run python examples/single_cell_sweep.py

# Save the full configuration plan after preparation.
uv run python examples/single_cell_sweep.py --write-plan /tmp/single-cell-plan.json

# Run on allocated GPUs, optionally with W&B tracking.
CUDA_VISIBLE_DEVICES=0,1 uv run --extra tracking python examples/single_cell_sweep.py \
  --run --gpus 2 --wandb-mode online --name single-cell-full-5k
```

Defaults:

- All three datasets, unconditional and conditional, seed 1: **132 training jobs,
  672 dependent inference jobs**. `--datasets`, `--modes`, and `--seeds` select subsets.
- 5,000 optimizer steps, up to 48,000 training cells (all available if fewer),
  global batch 512, learning rate 3e-4, EMA checkpoints, up to 2,000 held-out test cells.
- Masked, uniform CE, and uniform plug-in: no SC and hidden SC.
- Gaussian, spherical, and simplex continuous-only: hidden SC, matching the
  recorded results (there are no continuous-only no-SC rows there).
- Each continuous family paired with masked and uniform views: no SC and hidden SC.
- **SEDD: one plain absorbing score-entropy model, analytic sampler, SC disabled,
  token feedback disabled, no EB evaluations.** SEDD uses the matched backbone size.
- Regular sampling: 32 and 180 transitions. Continuous-only also uses 3 and 7
  transitions for exactly 4 and 8 NFE including final readout.
- Frozen EB bounds: 0.1 and 0.01. Uniform and uniform-hybrid models additionally
  use revisable EB. Budget stays at 180 transitions, not the longer gene count.
- Masked EB uses an optional greedy readout if genes remain after 180 steps,
  for at most 181 calls. This avoids an incomplete-cell exception on long gene
  sequences; the general sampler's default still raises at its budget.
- Continuous-only frozen EB remains the explicitly **experimental** commitment
  policy from the Sudoku report; it is not a native continuous sampler.
- Hidden SC probability 0.9. Token feedback disabled for every job.

The default backbone retains the Sudoku size (512 hidden, 8 blocks, 8 heads).
`--model-size paper` uses the draft's 256 hidden, 4 blocks, 4 heads, consistently
across all families. Microbatch/evaluation
batch size 1 for long gene sequences, with gradient accumulation preserving the
512-cell global training batch. Tune those with `--microbatch-size` and
`--eval-batch-size` after a memory pilot. Other changes are the dataset/task,
actual gene/vocabulary sizes, and distribution metrics. The draft itself uses
three seeds and usually 100 generated cells; choose `--seeds 1 2 3 --max-examples 100`
if that evaluation budget is preferred over the Sudoku budget.

`--run` requires prepared metadata and validates all tasks before starting Ray.
Each successful training job enqueues only its supported inference configurations.
The saved plan is fingerprinted; `--run --resume` requires the identical plan.
A new plan requires a new output name. The launch record in `core/outputs/scCANDI-launch/` contains dataset metadata,
GPU pilot measurements, and online W&B validation logs for the actual launch.

## Implementation validation

Synthetic sparse AnnData fixtures cover split isolation, training-only vocabulary
fitting, overflow, gene reordering, conditioning masks, and invalid counts/tokens.
Every one of the 22 model variants passed CPU loss/backward and generation checks
with small backbones, including each distinct sampler/commitment policy. A one-step
CPU training run also exercised the callback, saved checkpoint, and test-set
evaluation lifecycle. Together with the self-conditioning regression tests: 41
passed, 8 skipped (existing skip conditions). Ruff and diff whitespace checks passed.
Full-source preprocessing and production-size GPU memory/quality remain untested.

The sweep defaults to the W&B project `scCANDI`; enable `--wandb-mode online`
and select the account/team with `--wandb-entity`. Standalone checkpoint
evaluations publish scalar distribution metrics and NFE in separate W&B runs,
with the same group and a `parent_training_name` linking to training. Full gene
vectors stay in local `results.json` files. Training keeps final/last and best
checkpoints, with last saved after every epoch and without intermediate numbered
checkpoints. Training metrics are logged every optimizer step. Full validation
loss and generation metrics on 128 fixed validation cells run every 100 epochs. The
callback excludes full gene-vector records to bound repeated artifact storage;
final test evaluations still retain complete records.

The draft Appendix C refers to tiny CANDI defaults for the remaining settings.
The legacy config at commit `9be8bff` sets batch size 512 and optimizer LR 3e-4;
these are now explicit sweep defaults (`--batch-size`, `--learning-rate`).
`--eval-every-epochs` defaults to 100 for both validation loss and generation.
Short final accumulation windows cover the complete diffusion time range and
normalize gradients by their actual microbatch count; logged losses stay raw.

# scCANDI sweep launch — October 6, 2026

W&B project: https://wandb.ai/rz-text-dlp/scCANDI

Group: `single-cell-sudoku-matched-5k-20261006-eval100`.

Fresh restart requested by the user on October 6, 2026. The original group is
retained as historical interrupted runs; its in-memory progress was discarded
as requested. The prior per-epoch evaluation schedule is superseded by the requested
100-epoch cadence below.

Status at handoff: six training workers are active on GPUs 1–6 with online W&B logging; initial
optimizer-step losses are finite. This is a launch snapshot, not a live
monitor; consult W&B and the driver log for current status.

The grid contains 132 training jobs and 672 dependent evaluation jobs: three
single-cell datasets, unconditional and cell-type-conditioned tasks, 22 variants,
seed 1. Hidden SC and plain baselines are retained; token feedback is disabled.
SEDD is plain absorbing score entropy with analytic sampling only, no SC or EB.

Production settings: 5,000 optimizer steps, global batch 512, learning rate 3e-4, microbatch 2,
256 accumulation steps, evaluation batch 2, original Sudoku backbone size.
Training metrics log every optimizer step. Full validation loss and generation
metrics run after every 100 epochs. Generation uses a fixed, seeded subset of 128
validation cells, not the final test split. Evaluation callback files/artifacts
retain scalar metrics rather than full gene vectors; final test evaluations
retain their complete sample records. Training uses up to 48,000 cells. The evaluation cap is
2,000 cells; PBMC3K has 528 available test cells. The last checkpoint is saved each epoch and the best validation-loss checkpoint
is retained; intermediate numbered checkpoints are disabled to limit disk usage.

| Dataset | Genes | Vocabulary including overflow | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|
| PBMC3K | 8,573 | 367 | 1,899 | 211 | 528 |
| Dentate Gyrus | 17,002 | 744 | 13,113 | 1,457 | 3,643 |
| PBMC COVID | 2,000 | 29,815 | 310,338 | 34,482 | 73,807 |

Counts remain raw integer counts, with a training-derived overflow token. No
optional count cap was applied. The six source files were fetched by HTTP range
from the published CFGen ZIP and each extracted file passed its ZIP CRC check.
The whole archive MD5 was not checked because unrelated archive members were
not downloaded. Gene order, label mapping and split metadata are in the cache.

Launch script: `core/outputs/scCANDI-launch/run_sweep.sh`.
Launch PID/metadata: `core/outputs/scCANDI-launch/launch.json`.
Driver log: `core/outputs/scCANDI-launch/sweep.log`.
Durable Ray outputs: `/data/remote_cache/patrick/scCANDI-20261006/ray/`.
Ray worker temporary outputs: `/data/remote_cache/patrick/sc-ray/`.
GPUs: physical IDs 1–6. Existing work on GPUs 0 and 7 is not modified.

Online validation, separate from the scientific sweep:

- One-step real PBMC3K SEDD training: https://wandb.ai/rz-text-dlp/scCANDI/runs/2ld976xs
- Its checkpoint evaluation: https://wandb.ai/rz-text-dlp/scCANDI/runs/njm2eaqu

The online checks confirmed loss, MMD, Wasserstein and NFE metrics arrive in W&B.
Full-size GPU pilots passed for hidden-SC masked, hidden-SC simplex and plain SEDD
on all three datasets. The largest observed two-cell pilot allocation was about
15.9 GiB. These are setup/feasibility checks, not trained model quality results.
Full-gene training at this budget takes substantially longer than the Sudoku grid.

Batch size and LR follow the tiny CANDI defaults referenced by Appendix C:
legacy `9be8bff:configs/config.yaml` specifies 512 and 3e-4. Short final gradient
accumulation windows now stratify over the full time range and normalize by
their actual size, fixing the systematic epoch-boundary loss bias.
PBMC COVID reaches fewer than 100 epochs in 5,000 steps, so it receives final
test evaluation but no periodic validation at this requested cadence.

Restart validation: partial-accumulation tests cover full and short windows
across two epochs, asserting complete time strata, correct gradients, and raw
logged loss. Epoch-evaluation tests check both every-epoch and delayed cadence.
All targeted training/single-cell checks passed (one existing skip); Ruff and
whitespace checks passed. All 132 generated configs and six online W&B run
configs were checked for batch 512, LR 3e-4, and both 100-epoch intervals.

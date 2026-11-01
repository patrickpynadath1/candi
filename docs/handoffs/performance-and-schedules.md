# Performance and learned-schedule data handoff

Inventory date: 2026-11-01. This is a location/provenance guide, not a live job monitor.

## Start here

- Completed-score index: [CSV](performance-index-2026-11-01.csv) and [JSON](performance-index-2026-11-01.json). Generated from final result artifacts, not W&B summaries. JSON records the exact UTC snapshot time and campaign counts. Rows include final and validation scores, selected sampler/hyperparameters, evaluation seed scores, checkpoint and config paths, and result paths.
- Active development checkout: `/tmp/candi-distillation-recipe`, branch `dev-local-to-cluster`. There are unrelated uncommitted changes here: do not reset or stage everything. The checkout `/home/patrick/.codex/worktrees/54f2/duo` is a separate dirty worktree and supplies the shared environment.
- Read `/tmp/candi-distillation-recipe/AGENTS.md` for operational history, launch conventions, and migration requirements. Older status paragraphs there are historical snapshots; use current files for status.

## Storage roots

`/home/patrick/.cache/duo` resolves to `/data/remote_cache/patrick/duo`. These are aliases for the same data, not two copies. Serialized configs use both names. Preserve both the mount and alias during migration.

Under that root:

| Directory | Contents |
| --- | --- |
| `plans/<campaign>/` | Submitted manifests, resolved configs, source snapshots and provenance/reuse records |
| `nested-tuning/<output-campaign>/` | Training checkpoints, training metrics, Optuna studies, generated evaluation artifacts, selected settings and final scores |
| `data_cache/` | Task dataset caches; inspect resolved configs for dependencies outside this directory |
| `ray/` | Runtime/session logs; not the canonical performance store |
| `wandb-data/` and per-run `wandb/` | Tracking artifacts and logs |

Plots and analysis data live **outside the repository and experiment cache**, at:

`/home/patrick/.codex/visualizations/2026/10/11/01a12d34-6656-78b0-b299-814261a77f1e/`

These need separate preservation when moving development locally.

## Campaign map

All paths in the next table are relative to `/home/patrick/.cache/duo`.

| Campaign / plan name | Output under `nested-tuning/` | Scope |
| --- | --- | --- |
| `feedback-depth-20k-20261028` | Same name | Original 42 cases: Gaussian embeddings, masked, uniform posterior CE, hybrid; feedback/depth ablations on Sudoku Hard, text8, TinyStories. Three results reused; see plan `reused-results.json`. |
| `feedback-depth-cdcd-20k-20261030` | Same name | 18 reruns of Gaussian/hybrid using new CDCD time sampling and the inverse-CDF clamp fix. Prefer these over old adaptive Gaussian/hybrid results. |
| `feedback-geometry-cdcd-20k-20261030` | **`feedback-depth-cdcd-20k-20261030`** | 18 cases: simplex, spherical geodesic, spherical-vMF × no SC / expected-embedding SC × three tasks. Separate manifest, shared output root. |
| `feedback-renoising-128nfe-20261031` | Same name | 48 inference-only cases using existing checkpoints, with sampled-token and greedy-token forward-renoising policies. No retraining. |

At this inventory, all four campaigns are complete: original **42/42** (39 local finals + 3 reused), Gaussian/hybrid CDCD **18/18**, geometry **18/18**, and renoising **48/48**. The index resolves reused results to their source files and flags those rows explicitly. These are logical case counts; they are not 126 independently trained models, because reruns and inference-only evaluations reuse or supersede cases.

The renoising plan includes `origins.json`, `source-provenance.json`, and `pending-unfinished.json`. The last records the 12 geometry text8/TinyStories cases unavailable when that sweep launched. **Those were excluded, not automatically appended when training later finished.** Its 48/48 completion does not cover those 12.

Each plan's `manifest.json` is the membership authority. Do not count the whole shared CDCD output root as the 18-case geometry campaign.

W&B projects generally use `rz-text-dlp/<output-campaign>`, e.g. `https://wandb.ai/rz-text-dlp/feedback-renoising-128nfe-20261031`. Local artifacts are the primary source of truth; transient W&B connection-reset errors occurred during otherwise continuing evaluations.

## Reading a case

Typical case name: `text8-gaussian_embedding-expected_embedding-seed-1`.

```
<output-root>/<case>/
  config.yaml
  protocol.json
  training/candidates/<training-hash>/
    trained.json
    training/
      config.yaml
      evaluation-config.yaml
      checkpoints/last.ckpt
      checkpoints/best.ckpt                  # when present
      metrics/version_*/metrics.csv
    inference/<policy>/
      optuna.sqlite
      selected.json
      candidates/<inference-hash>/
        result.json
        seed-*/...
  final/
    result.json
    seed-1041/config.yaml
    seed-1041/results.json
    seed-1041/wandb/...
    seed-1042/...
```

Use paths recorded in JSON rather than assuming a particular hash or checkpoint name. Reused and inference-only cases can reference a checkpoint in another campaign.

- **`final/result.json.score`**: final held-out evaluation score, aggregated over evaluation seeds.
- **`final/result.json.seed_scores`**: individual final evaluation scores. These are not independent training seeds.
- **`selection.score` / `selection.seed_scores`**: validation results used for selection. Do not report these as test results.
- **`selection.policy`**: selected sampler configuration and search definition.
- **`selection.inference_parameters`**: chosen temperature and any other tuned parameters.
- **`selection.checkpoint`**: exact checkpoint and configuration paths used.
- **`selection.training_parameters`**: outer-loop training parameters, where applicable.
- **`inference/<policy>/selected.json`**: per-sampler selection record. Inspect these and candidate `result.json` files when comparing samplers; the final file reports the overall selected winner only.
- **`optuna.sqlite`**: trial history/state. Preserve journal/WAL sidecars and use a consistent SQLite backup if copying live studies.
- **Per-seed `results.json`**: detailed task metrics. Sample text may be embedded or stored in task-specific artifacts/W&B tables; inspect the actual file/config before assuming a separate sample file exists.

Metric meanings: Sudoku score is solved-puzzle accuracy (multiply by 100 for percent); TinyStories uses reference BLEU, not judge scores; text8 score here is mean unique valid generated words of length **at least 5 per sample**, not a percentage. Detailed text8 metrics also include length ≥6 and other validity/diversity statistics. Read `protocol.json` for the exact objective and splits.

These main comparisons use one training seed, 20k updates, and 128-NFE inference with validation-selected parameters. `none`, `expected_embedding`, `hidden`, `double_depth` distinguish no SC, expected-embedding SC, hidden-state SC, and twice the backbone depth without SC. Expected-embedding and hidden-state SC are different methods.

## Schedule plots and source data

All directories below are relative to the visualization root given above. **Always read `metadata.json` for checkpoint paths/steps; a plot does not update when training advances.**

| Directory | What to use / limitations |
| --- | --- |
| `cdcd-final-three-tasks/` | `schedules.png`, PDF, `curves.csv`, metadata and `plot.py`: final 20k Gaussian/hybrid no-SC and expected-embedding-SC schedules across Sudoku Hard, text8, TinyStories. Primary final Gaussian schedule comparison. |
| `simplex-final-comparison/` | `schedules.png`, `metadata.json`: final 20k simplex schedules with/without SC for Sudoku and text8. Shows physical time and clean coefficient α. Metadata includes concentration at selected times. Generation script is also preserved with this handoff. |
| `cdcd-sudoku-final/` | Final Gaussian/hybrid Sudoku sigma plot, CSV, metadata, script. |
| `geometry-schedules/` | Native simplex/geodesic/vMF schedule plots and CSV. **Older snapshots:** vMF Sudoku and simplex text8 were intermediate checkpoints. Do not present these as the now-completed models. |
| `geometry-empirical/` | Empirical corruption probability and normalized rank degradation using actual geometry forward kernels; CSV, metadata, simulation script and plots. Inherits the older geometry checkpoint selection. |
| `geometry-gaussian-empirical/` | Adds final Gaussian/hybrid curves; `combined-curves.csv`. Preferred requested Sudoku layout is **`sudoku-combined-panels.png`** (or PDF): separate method panels in one figure. Geometry portions still carry the earlier checkpoint caveat. |
| `gaussian-text-empirical/` | `gaussian_embedding.png`, PDF, `curves.csv`, metadata, simulation/plot scripts: final Gaussian text8/TinyStories empirical token corruption and rank degradation. |
| `gaussian-loss-bins/` | `loss-bins.png`, PDF, `bins.csv`, metadata and `measure.py`: fresh validation CE measurements by global-time bin vs learned CDCD loss fit for final Gaussian text tasks. |
| `gaussian-mixture-fit/` | `mixture-fit.png`, PDF, `fit-curves.csv`, `summary.json`, `fit.py`: fits combinations of normalized empirical token error and rank degradation to global time. Includes constant mixture fits and pointwise required weights. |
| `gaussian-schedules/`, `hybrid-schedules/`, `cdcd-schedules/` | Historical/adaptive or earlier CDCD comparisons. Preserve for before/after analysis; inspect metadata rather than treating them as current. |

Gaussian σ, simplex α/concentration, and spherical geodesic α/vMF κ are different quantities. Native parameter plots alone do not put corruption severity on a shared scale.

Empirical plots use uniformly sampled clean tokens across the full model vocabulary, actual forward kernels, 8,192 samples per point, and normalized midrank `(rank - 1)/(V - 1)`. Random-noise error baseline is `1 - 1/V`; normalized rank baseline is 0.5. Thus the curves depend on vocabulary and the learned representation, and are not corpus-frequency-weighted. Hybrid curves in these analyses measure the **Gaussian view only**, not joint information including visible discrete tokens.

Schedule extraction uses raw checkpoint state, not EMA model weights. The scripts instantiate the saved time sampler and load `algorithm.time_sampler.*` from checkpoint `state_dict`, then apply the corresponding corruption schedule. Preserve that distinction when comparing with evaluation settings.

CDCD fits a monotone approximation to expected loss as a function of physical time and inverts it. It does **not** directly aim to linearize token corruption probability or rank degradation. The loss-bin analysis compares measured loss to that fitted loss, not to an integral of the loss.

## Other experiment archives

Under both `plans/` and `nested-tuning/` unless noted:

- `masked-centered-mlp-sudoku-20k-20261030`: completed normal-MLP vs gated mean-subtraction ablation.
- `masked-attention-sudoku-20k-20261030`: earlier expanded-attention replacement experiment, stopped at user request; do not restart.
- `mdlm-depth-mixing-sudoku-20k-20261031`: separate newer architecture workstream; inspect its manifest/logs independently. It is not part of the two sweeps above.
- `insertion-before-after-text8-20261028`, `insertion-crossover-text8-20261028`, `insertion-policies-text8-20261028`: smaller 2k-update insertion experiments. Crossover/policies plan `comparison.json` links reused evaluations.
- `plans/feedback-depth-cdcd-sudoku-extreme-20261030/`: separate Sudoku Extreme plan; do not conflate with Sudoku Hard.
- Older `nested-tuning/` directories include `three-tasks-20261013`, `adaptive-three-tasks-20261014`, `hybrid-rank-three-tasks-20261022`, `posterior-mean-three-tasks-20261022`, `depth-feedback-three-tasks-20261022`, `gaussian-sc-gradients-20261022`, and distillation/alpha campaigns. These are archival, not matched replacements for the current 20k/CDCD comparisons.

## Reproduction and preservation

Use the launch records and pinned plan source to reproduce an existing job. The current working tree can differ from the snapshot used to train it. The Gaussian CDCD rerun's corrected source is `plans/feedback-depth-cdcd-20k-20261030/source-clamp-fix/`; geometry/renoising have their own source snapshots. Do not infer source equivalence solely from branch names.

Use the shared uv environment for ML inspection:

```bash
cd /tmp/candi-distillation-recipe
UV_PROJECT_ENVIRONMENT=/home/patrick/.codex/worktrees/54f2/duo/.venv \
  /home/patrick/.local/bin/uv run --locked \
  --extra tuning --extra scoring --extra tracking python SCRIPT.py
```

Preserve complete case trees, plans/source snapshots, reuse references, protocol/config files, Optuna databases, checkpoints, and visualization metadata/CSVs. Do not clean up a checkpoint just because training ended: inference-only results and analysis scripts reference checkpoints across campaigns. Do not blanket-stop Ray; other jobs/clusters coexist on the server.

Refresh the compact performance index with `python docs/handoffs/refresh_performance_index.py` from this checkout. This only reads saved artifacts and rewrites the dated index files; it does not query W&B, start jobs, or mutate campaign data.

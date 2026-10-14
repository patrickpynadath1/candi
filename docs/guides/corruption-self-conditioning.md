# Corruption × self-conditioning on Sudoku

`examples/corruption_self_conditioning_sweep.py` sets up six corruption recipes,
hidden self-conditioning off/on, and five training seeds (1–5): **60 training
runs at 10,000 steps**, followed by **180 inference jobs**. Preview is the default;
it does not initialize Ray or prepare data.

Training logs online to W&B project `self-cond-sweep`, entity `rz-text-dlp`.
Use `--wandb-project` and `--wandb-entity` to override these. Run names identify
the corruption, self-conditioning setting and seed; the experiment name is
the W&B group. Periodic sampler curves and separate final evaluation runs also log to W&B.
Final evaluation run names include the sampler policy; reports are saved locally too.

```bash
uv run python examples/corruption_self_conditioning_sweep.py
uv run python examples/corruption_self_conditioning_sweep.py --write-plan /tmp/corruption-plan.json
uv run python examples/corruption_self_conditioning_sweep.py --run
uv run python examples/corruption_self_conditioning_sweep.py --run --resume
uv run python examples/corruption_self_conditioning_sweep.py --report
```

Launch uses one GPU per job, with at most two simultaneous jobs by default.
Use `CUDA_VISIBLE_DEVICES` to restrict devices, `--max-concurrent` to adjust
concurrency, and `--data-cache-dir` for an existing Sudoku cache. The default
cache root is `~/.cache/duo/data_cache`; launch requires the matching cache to
exist and does not generate a new dataset. Resume requires
the original arguments and an identical plan fingerprint. `--name` and
`--storage` identify the experiment directory; defaults are
`~/.cache/duo/outputs/ray/corruption-sc-10k`. Ray's temporary worker logs and
artifacts also live on the cache drive (`--ray-temp-dir`, default
`~/.cache/duo/ray`). W&B run files are inside each trial output; artifact staging
defaults to `~/.cache/duo/wandb-data` unless `WANDB_DATA_DIR` is already set.

The existing `self-cond-sweep-v2` was migrated to
`~/.cache/duo/worktrees/54f2/outputs/ray/self-cond-sweep-v2`. Its original path is
a symlink so saved checkpoint references remain valid. For that sweep, pass
`--storage ~/.cache/duo/worktrees/54f2/outputs/ray --name self-cond-sweep-v2`.
Migration alone does not restart training.

## What varies

| Recipe | Schedule / representation |
| --- | --- |
| Blurring | Blurring schedule, feature correlation 0.5, embedding view |
| Gaussian | One-hot linear rank schedule, analytic inversion, CDCD adaptation |
| Spherical | Native spherical default, automatic truncation disabled |
| Simplex | Default constant-linear schedule and simplex view |
| Masked | Log-linear absorbing corruption, token view |
| Uniform | Log-linear uniform corruption, posterior token view |

Gaussian is a single schedule choice. Its CDCD clock observes cross-entropy
after 1,000 warmup steps, using the existing adaptive schedule defaults.
Adaptation is recorded in every run's metadata and saved in the checkpoint.
This experiment compares **corruption recipes**: representations and Gaussian
adaptation remain part of the recipe. It cannot attribute a Gaussian gain to
corruption geometry independently of adaptation.

Every recipe uses the same single-view HybridDiffusion assembly, posterior CE
objective, identity time conditioning, backbone, optimizer, BF16 mixed precision,
48,000 hard Sudoku training examples, batch size 256, and EMA evaluation.
Self-conditioning uses hidden feedback, with training probability 0.9 when
enabled. The two SC settings use matching seed IDs. Every 1,000 optimizer updates, all three samplers evaluate the same fixed
128-puzzle subset with EMA and isolated Torch RNG. Metrics appear under
`sudoku/<policy>/valid_solution_rate` in W&B. A rolling training checkpoint
is saved every 1,000 updates (only the latest periodic checkpoint is retained).
Use `--eval-every` and `--periodic-eval-examples` to adjust diagnostics. Insertion/deletion, block diffusion and token feedback
are not enabled in this fixed-length comparison. Shared CE means these are
controlled component combinations, not reproductions of each paper's full recipe.

Ten thousand steps fixes optimizer updates and examples seen; SC has extra
training forward-pass cost. Five seeds are a practical initial replication
budget, not a power guarantee for a particular effect size.

## Inference and choosing the winner

Each final checkpoint runs three policies:

- `native_128`: 127 reverse transitions plus one greedy final readout, exactly
  128 network evaluations. Uses each view's native reverse update. Spherical
  uses manifold flow because it does not provide an ancestral kernel.
- `native_128_greedy`: the same 127 native transitions and final readout, but
  each transition receives a one-hot top-1 clean-token estimate instead of the
  full posterior. Native transition noise is otherwise unchanged.
- `eb_greedy`: frozen entropy-bounded sampling, greedy clean estimates, entropy
  bound 0.1, and at most 128 network evaluations. It may finish early.
  Continuous-view EB is an experimental commitment policy.

All jobs evaluate the same 2,000 validation puzzles with inference seed 1729.
A fixed shuffled split (seed 271828) reserves 1,000 puzzles for sampler
selection and 1,000 for reporting. For each corruption × SC configuration,
choose the sampler with the highest mean selection solve rate across all five
seeds. Ties prefer lower mean NFE, then policy name. The selected policy is
shared across seeds within that configuration; configurations choose separately.
Reported scores use only the reporting half. No checkpoint selection is done:
every run uses the final 10,000-step checkpoint.

`--report` produces `analysis/all-policies.csv`, `selected-per-seed.csv`,
`summary.csv`, and `status.json`. It checks configuration metadata, training
step, EMA, puzzle identities and NFE. Incomplete configurations have no winner.
The summary includes mean solve rate and across-seed standard deviation in
percentage points; it does not claim statistical significance. Per-seed and
per-puzzle outputs remain available for paired comparisons and interaction
analysis. Treat training seeds as replications rather than counting puzzles
as independent training runs. Retain all policies when checking whether a
conclusion depends on sampler selection.

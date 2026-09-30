# Simplex Sudoku experiments

Run these entry points from the repository root using its Python environment.
Training, data audits, and evaluation diagnostics log to W&B. Run artifacts belong
under an explicit output directory, not beside source files.

| Entry point | Purpose |
| --- | --- |
| `prepare_data.py` | Audit an existing Sudoku cache and convert it to the paper layout. |
| `reproduce.py` | Plan, train, evaluate, or aggregate the five-seed experiment matrix. |
| `audit_sampler.py` | Compare current and EMA checkpoint weights using identical examples and seeds. |
| `smoke.sh` | Exercise simplex, masked, and uniform training with self-conditioning off/on. |

```bash
# Materialize commands only; no training is launched.
python scripts/simplex/reproduce.py --full --phase plan --output runs/simplex-paper

# Launch a NEW queue only when no other queue owns these GPUs/output directories.
python scripts/simplex/reproduce.py --full --phase run --gpus 1 2 \
  --output runs/simplex-paper --weights current --entity YOUR_WANDB_ENTITY

# Summarize completed 5,000-puzzle evaluations.
python scripts/simplex/reproduce.py --phase summarize --output runs/simplex-paper

# Compare current and EMA weights from one checkpoint.
python scripts/simplex/audit_sampler.py \
  --config runs/EXPERIMENT/.hydra/config.yaml \
  --checkpoint runs/EXPERIMENT/checkpoints/last.ckpt \
  --output runs/EXPERIMENT/audit/comparison.json

PYTHON_BIN=python OUTPUT="$PWD/runs/simplex-smoke" bash scripts/simplex/smoke.sh
```

`--weights ema` is the paper protocol and launcher default. `--weights current`
sets `eval.disable_ema=true` for training-time validation, generation, and final
evaluation. Checkpoint saving is controlled independently by
`checkpointing.save_final`; the Sudoku experiment enables it.

The launcher is not a persistent job scheduler: it does not skip completed jobs
or archive old logs when rerun. Check existing queue metadata before launching.
The already-running reproduction uses an external frozen source snapshot and is
unaffected by changes to these workspace scripts.

See [the implementation record](../../papers/methods/simplex/implementation.md)
for paper mappings, validation evidence, and experiment status.

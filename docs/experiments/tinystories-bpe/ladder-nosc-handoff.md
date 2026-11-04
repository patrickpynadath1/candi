# Task: run the no-SC TinyStories BPE tokenizer ladder

Run the no-self-conditioning (no-SC) versions of the tokenizer ladder cases from `portable-v2-mi210`. You have approval to build the manifest and launch. No need to check back before submitting.

## Cases (12 training jobs, seed 1)

| Path | Sweep and overrides | Vocab |
|---|---|---|
| Gaussian learned embeddings | `feedback_cdcd`, `ablation_path=gaussian`, `feedback=none` | 260, 500, 1000, 5000 |
| Masked | `feedback`, `ablation_path=masked`, `feedback=none`, `+portable_objective=masked_elbo` | 260, 500, 1000, 5000 |
| Uniform posterior CE | `feedback`, `ablation_path=uniform`, `feedback=none` | 260, 500, 1000, 5000 |

Case names: `tinystories-bpe-{V}-{gaussian_embedding,masked,uniform_posterior_ce}-none-seed-1`.

Do not rerun vocab 10k. Those no-SC cases already exist in `portable-v2-mi210`. Launch the Gaussian cases first.

## How
1. In `build_manifest.py`, add a `ladder_nosc()` next to `ladder()`. It is a copy of `ladder()`'s three diffusion entries (Gaussian, masked, uniform) with `feedback=none` in place of `feedback=expected_embedding`. Leave out AR. Do not change `ladder()`.
2. Build a manifest from `ladder_nosc()` for vocab 260/500/1000/5000, using the same `CAPACITY` (260 → 1024, 500 → 768, 1000/5000 → 512).
3. Write it to a new plan, `portable-v2-mi210-ladder-nosc`, with its own output root and W&B project.
4. Launch it with the same runner and settings as `portable-v2-mi210` (`nested_tuning`, inference concurrency 1 per case).

## Keep identical to `portable-v2-mi210`
Only `feedback` changes. Everything else is the same as the SC ladder cases:
- Source commit `aa4ee07` plus the same `source.patch`, and the same torch 2.9.1+rocm6.3 env.
- The same `partitions.json` (seed 7351), `allowed_ids.json` cohort, tuning (128) and test (256) indices, and tokenizer files.
- Training: 20k updates, batch 256, LR 3e-4, width 512, 8 layers, and the same warmup, EMA, precision and compile settings.
- Sampler tuning: the same policies and search budget, 128 NFE, corpus BLEU as the tuning metric, validation seeds 41/42, final seeds 1041/1042.

## Do not
- Touch the running `portable-v2-mi210` campaign: its manifest, plan, outputs or Ray job.

## When done
Package the results the same way as `portable-v2-mi210` (`package_results.py`): one `.tar.gz` named `portable-v2-mi210-ladder-nosc-<date>.tar.gz`, with `README.md`, `index.json`, `summary.csv`, `provenance/` (manifest, `build_manifest.py`, source patch, env freeze) and the per-case result trees under `cases/`. No checkpoints, W&B caches, datasets or credentials. That archive is the deliverable; it will be shared as is.

# E2D2 encoder/decoder scaling

The Hydra grid in `configs/sweep/e2d2_depth_scaling.yaml` prepares 16 runs:
text8 and TinyStories, encoder/decoder arms, and variable depth 1/2/4/8.
The other component stays at 8 layers. The 8/8 control occurs twice per task,
with the same seed (1); these are duplicate controls, not independent seeds.
A matched shared-transformer block-diffusion baseline needs two additional
20k-step runs; the stopped 2k-step text8 pilot is only a functionality check.

All runs use masked MDLM ELBO-weighted cross entropy, width512, 8 heads,
block size32, no self-conditioning, 20k optimizer updates, batch256 (microbatch8,
32 accumulation steps), warmup2500, lr0.0003, BF16, torch.compile and activation
checkpointing. Text8 uses length256; TinyStories uses length1024 bytes.
The E2D2 wrapper supplies cached final encoder representations to the decoder.

Prepare with the project uv environment (this does not train):

```bash
uv run --locked --extra tuning --extra scoring --extra tracking \
  python -m core.sweeps.prepare -m +sweep=e2d2_depth_scaling
uv run --locked --extra tuning --extra scoring --extra tracking \
  python -m core.sweeps.prepare collect \
  /home/patrick/.cache/duo/plans/e2d2-depth-scaling-20k-20261101
```

The exported manifest is consumed by the existing Ray nested-search runner.
After approval to launch, with GPUs5/7 still available:

```bash
CUDA_VISIBLE_DEVICES=5,7 uv run --locked --extra tuning --extra scoring --extra tracking \
  python -m core.sweeps.nested_tuning \
  /home/patrick/.cache/duo/plans/e2d2-depth-scaling-20k-20261101/manifest.json \
  --output /home/patrick/.cache/duo/nested-tuning/e2d2-depth-scaling-20k-20261101 \
  --address local --case-concurrency 2 --concurrency 1
```

Inference tuning uses separate validation/test indices inherited from the
existing task manifests. Native ancestral sampling tunes temperature0.5–1.5
(16 trials); native top-1 tests terminal greedy/ancestral decoding (2 trials).
Each policy retains its own validation-selected winner for final testing.
The total budget is128 NFE per sequence: conservatively16 per text8 block and
4 per TinyStories block. Fully prompted blocks cost no NFE, so actual usage
can be lower. Entropy-bounded sampling is omitted here because it can fail to
finish a32-token block within4 NFE. This is an architecture comparison under
fixed inference settings/budgets, not a full sampler comparison.

The nested runner chains training, tuning and held-out evaluation. Results belong
under the cache output above; W&B project is `e2d2-depth-scaling-20k-20261101`.
No training was launched while preparing this plan.

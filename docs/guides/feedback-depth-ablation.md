# Feedback versus depth ablation

The native Hydra grids are `+sweep=feedback` and `+sweep=feedback_hybrid`.
The first expands three pure paths × four feedback settings × three tasks (36
cases). The second expands Gaussian-plus-masked × two feedback settings × three
tasks (6 cases). No bespoke configuration expansion script is used.

| Path | No feedback | Expected embedding | Hidden representation | Double depth, no feedback |
| --- | --- | --- | --- | --- |
| Gaussian learned embeddings, adaptive clock | Yes | Yes | Yes | Yes |
| Gaussian learned embeddings + masked, adaptive clock | Yes | Yes | No | No |
| Masked | Yes | Yes | Yes | Yes |
| Uniform posterior CE | Yes | Yes | Yes | Yes |

Shared settings: width 512, eight heads, eight layers (16 for double depth),
batch 256, 2,500 warmup updates, 20,000 updates, BF16, compiled backbone,
one training seed. Gaussian embeddings are trainable. Feedback has probability
0.9 during training and 1 during evaluation; expected embeddings use
`posterior_mean`, hidden feedback uses `hidden`, with detached first-pass states.

Tasks: Sudoku hard, text8 and TinyStories. Inference keeps 128 NFE, includes
native, native greedy and entropy-bounded greedy policies, and uses the existing
validation-only Optuna searches. Native temperature and EB temperature/entropy
bound are tuned separately. Final generation seeds and held-out indices differ
from tuning seeds/indices. Task scores remain Sudoku accuracy, the configured
text8 word metric, and TinyStories reference BLEU.

Preparation (these commands export configs, not training jobs):

```bash
uv run --locked --extra tuning --extra scoring --extra tracking \
  python -m core.sweeps.prepare -m +sweep=feedback
uv run --locked --extra tuning --extra scoring --extra tracking \
  python -m core.sweeps.prepare -m +sweep=feedback_hybrid
uv run --locked --extra tuning --extra scoring --extra tracking \
  python -m core.sweeps.prepare collect \
  /home/patrick/.cache/duo/plans/feedback-depth-20k-20261028
```

The October 28 launch has three files under that plan directory:

- `manifest.json`: all 42 configured cases.
- `pending-manifest.json`: the 39 cases actually submitted.
- `reused-results.json`: checkpoint and final-result references for three matching
  completed Sudoku cases from `local-diffusion-20k-20261027`.

Reused cases are Gaussian expected-embedding SC and Gaussian-plus-masked with
and without expected-embedding SC. Model, training, data, callback/sampler settings
and complete tuning specifications were compared before reuse (excluding the
callback output directory). Their prior W&B runs remain in the prior project;
the reference file makes that provenance explicit. Do not submit the full
manifest unless intentionally retraining these three cases.

The new run uses GPUs 0,1,2,5, four concurrent cases, the existing nested Ray
runner, and W&B project `rz-text-dlp/feedback-depth-20k-20261028`. Results,
`driver.log`, `driver.pid`, the exact launch command/environment in `launch.json`,
and reuse references live under
`/home/patrick/.cache/duo/nested-tuning/feedback-depth-20k-20261028`.

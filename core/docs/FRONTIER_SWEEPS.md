# Frontier sweeps

`uv run python frontier.py` evaluates a Cartesian grid using the
normal `main.py` entrypoint. It loads pretrained weights; it does not train.
Each point runs in a fresh process and uses the same task metrics as standalone
evaluation. GPU selection is inherited from `CUDA_VISIBLE_DEVICES`.

## OpenWebText

Use the same architecture, token space, schedule, and modifiers as the checkpoint.
For an original MDLM checkpoint trained with the small 768-wide, 12-layer DiT:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python frontier.py \
  +experiment=openwebtext frontier=openwebtext sampler=ancestral \
  model.adaLN=true \
  evaluation.checkpoint=/absolute/path/to/pretrained.ckpt \
  evaluation.legacy=true noise_schedule.epsilon=0 \
  task.num_eval_samples=128 loader.eval_batch_size=4 \
  'frontier.axes=[{parameter:parameterization.temperature,values:[0.8,1.0]},{parameter:sampler.steps,values:[32,64]}]' \
  hydra.run.dir=core/outputs/frontiers/openwebtext_mdlm
```

For native core checkpoints leave `evaluation.legacy=false`. CANDI imports use
`evaluation.legacy=candi` and the matching hybrid experiment/architecture; the
sweep runner makes no algorithm-specific changes to checkpoint loading.

The OpenWebText preset maximizes empirical token entropy and minimizes
GPT-2-large generative perplexity, selecting a separate temperature frontier
within each seed and requested step budget. `mean_nfe` records actual model
calls, including final denoising, rather than assuming it equals `sampler.steps`.

GPT-2 uses the same ID for BOS and EOS. The scorer now permits that ID at the
start of a document and stops at the first subsequent EOS. This fixes the legacy
case where every BOS-prefixed sample scored zero tokens. Set
`task.scorer.ignore_initial_bos=false` only when reproducing that old boundary
behavior. The shifted cross-entropy, corpus weighting, and entropy equations
remain unchanged. This boundary correction can change historical perplexities.

## Text8 and Sudoku

Select `+experiment=text8 frontier=text8 sampler=ancestral` for Text8. Its
objectives are `generated/words/counts/unique` and
`generated/words/counts/valid`, both maximized. Length-filtered objectives are
also available, e.g. `generated/by_min_length/4/aggregate/counts/valid`.
Metric paths follow `core.evaluation.reporting.scalar_metrics` and use slashes.

The default `frontier=sudoku` varies posterior temperature and EB entropy bound,
maximizing accuracy and minimizing actual mean NFE. It requires an EB sampler
exposing `entropy_bound`. Choose axes appropriate to the selected sampler.

To compare quality, diversity, and compute together, configure three objectives:

```bash
'frontier.objectives={entropy:max,generative_perplexity:min,mean_nfe:min}' \
'frontier.group_by=[seed]'
```

Hydra dictionary overrides merge keys. When changing the objective set to
unrelated metrics, remove old keys with `~frontier.objectives.<key>` as needed,
or create a new frontier preset. Axes are a list and are replaced as a whole.
`frontier.seeds=[1,2,3]` repeats the grid with paired seeds. These are independent
per-seed frontiers, not confidence intervals or averages across seeds. All points
should use the same evaluation sample count for comparable word counts.

## Planning, restart, and results

- `frontier.dry_run=true` writes a manifest without loading a model or sampling.
  The checkpoint must exist, and axis names must refer to existing config fields.
- Reuse the same `hydra.run.dir` to restart. Completed points with matching
  resolved settings and checkpoint path/size/modification time are skipped.
  The cache does not hash checkpoint contents or code: use `frontier.resume=false`
  after implementation changes or deliberate in-place checkpoint replacement.
- Each point saves `config.yaml`, `results.json`, `evaluation.log`, and a completion
  marker under `points/<id>/`. Full task sample records remain in `results.json`.
- `frontiers.json` and `frontiers.csv` contain the active grid's scalar metrics,
  swept settings, seed, result paths, and Pareto membership. Equal points both
  survive; dominance requires a strict improvement in at least one objective.
- `frontier.svg` plots the first two objectives. Membership uses **all** objectives
  and respects `group_by`; the projection need not look like a 2D frontier when
  more than two objectives are configured. Hover over points to see their IDs.
- A failed point stops the runner, preserving completed results. Read its log,
  correct the problem, and restart. Summaries are updated after each success and
  can therefore be partial; `manifest.json` lists the full planned grid.

The runner is deliberately sequential on its assigned device(s). Use separate
output directories for concurrent jobs; do not point simultaneous runners at
the same directory. Ordinary Hydra training sweeps remain available via
`uv run python main.py --multirun ...`.

## Pretrained OpenWebText validation

The cached `openwebtext-train/mdlm_orig_1024/checkpoints/last.ckpt` was evaluated
using its 65,000-step EMA weights, a 768-wide 12-layer DiT with AdaLN, and the
cached GPT-2-large reference scorer. A single requested point used temperature
0.8, 32 ancestral steps plus final greedy denoising, seed 1, and 32 sequences of
length 1,024. Results: perplexity **30.7164**, mean token entropy **4.92239 nats**,
23,944 scored tokens, and 33 NFE. Evaluation took about 34 seconds on an A6000.
This verifies execution on a real pretrained language model; 32 samples at one
point do not establish a quality/diversity frontier or a large-sample benchmark.

Artifacts are in `core/outputs/frontiers/openwebtext_mdlm_pretrained/`:
`frontiers.json`, `frontiers.csv`, `frontier.svg`, and per-point generated text,
config, and logs. The successful point ID is `97030580c7d928f6`.

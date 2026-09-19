# HubPaths implementation check and example measurements

The task uses ordered lists of distinct waypoint nodes. Responses are continuous
walks whose first visits to those nodes follow prompt order. Dataset generation,
exact sampling/scoring, independent verification, AR/infilling adapters, training
evaluation, n-gram diagnostics, and small parameter sweeps are implemented under
`tasks/hubpaths`. See `tasks/hubpaths/README.md` for commands and all parameters.

## Language-like example

The saved graph has 10,000 nodes, gamma 2.1, three hierarchy levels, temperature 1,
and a global 77-step budget. Each prompt has three waypoints, making sequence
length 82. The example contains 2,048 training paths and 16 each in validation,
test, and OOD. Coverage top-ups were disabled (`min_edge_count=0`) to keep this an
exploratory report. This is not the full two-million-example production dataset.

Data: `data_cache/hubpaths/language_like_report/`.
Reports: `analysis_results/hubpaths/language_like/`.
All 2,096 saved sequences passed batched ordered-route verification.

Response-only statistics, excluding padding and prompt tokens:

| Order | Occurrences | Unique n-grams |
|---|---:|---:|
| 1 | 37,415 | 3,749 |
| 2 | 35,367 | 6,537 |
| 3 | 33,319 | 9,742 |
| 4 | 31,271 | 13,952 |
| 5 | 29,223 | 17,420 |

The fitted path rank-frequency slope is -1.04664 and unigram entropy is 8.16405
bits. The top 10 tokens account for 26.24% of all token occurrences; the top 50
account for 56.11%. Nineteen tokens individually account for at least 1% each.
Thirty-seven tokens account for half the corpus. There are 2,370 singleton node
tokens and 6,251 graph nodes unobserved in these training paths. Final goals and
intermediate waypoints are reused across starts for efficient sampling, so this
small corpus is not representative of all possible waypoint combinations.

The rank-frequency chart now labels actual ranks and occurrence counts on
logarithmically spaced axes. A previous tick labeled 0.893 represented
`log10(rank)=0.893`, or rank approximately 7.8, not a fractional node ID.

Exact entropy and counts of shortest routes are measured on 32 evenly spaced
training examples; n-gram counts cover **all** 2,048 training examples. The JSON
records the distinction. Token frequencies are also available as TSV; every exact
n-gram count is available in `ngram_counts.sqlite`.

## Tiny dataset and sweep

`data_cache/hubpaths/tiny_debug/` contains the default 200-node, three-waypoint
corpus. It has 2,041 train examples (2,000 base plus 41 top-ups), and 100 examples
in each evaluation split. Every undirected edge appears at least once in training;
no edge falls below this preset's requested coverage target.

`analysis_results/hubpaths/gamma_sweep/sweep.csv` records a four-value exploratory
sweep with 256 training examples per setting:

| Gamma | Rank-frequency slope | Mean steps | Nodes used | Mean conditional entropy (bits) |
|---|---:|---:|---:|---:|
| 2.1 | -1.4319 | 8.4961 | 148 | 12.4036 |
| 2.5 | -1.4824 | 9.6484 | 155 | 14.1320 |
| 3.0 | -1.3655 | 8.5352 | 170 | 13.3595 |
| 4.0 | -1.3822 | 8.9570 | 168 | 13.5618 |

The slope is not monotonic on this finite setup. No monotonicity guarantee is
imposed, and these small runs should not be interpreted as a fitted scaling law.

## Training checks

AR and MDLM each completed 300 optimizer steps on host GPU 2 (RTX A6000), outside
the filesystem sandbox. Both used the actual repository training entry point,
64-wide two-block models, batch size 16, and a separate 80-example smoke corpus
at `/tmp/hubpaths-smoke`. Checkpoints and evaluation records are saved under:

- `runs/hubpaths_smoke/ar/`
- `runs/hubpaths_smoke/mdlm/`

Both callbacks evaluated two test and two OOD prompts with 16 samples per prompt.
The runs reported zero valid held-out paths: they establish integration and finite
training loss, not successful task learning. Validation token NLL at step 300 was
4.83966 for AR and 3.76885 for MDLM; the latter is a stochastic bound estimate.
Undefined valid-path statistics are null in the final evaluator implementation.

## Automated checks

The HubPaths tests plus existing Sudoku and AR tests passed together (45 tests).
They cover brute-force distribution checks, sample-frequency chi-square checks,
high and low temperature, every original verifier failure reason, waypoint order,
10,000 randomized/corrupted sequences, all backbone topologies, split isolation,
byte-identical regeneration, streamed exact n-gram counts, batched suffix recursion,
and persisted-table loading. No tests assert mathematically invalid claims of
strict shortest-path-only support at positive temperature or monotone finite-sample
Zipf slopes. The low-temperature test checks a fixed sample, plus stable numerical
normalization through horizon 128.

The original pair interfaces remain available alongside the ordered-waypoint
interfaces. `require_simple` and random hint-selection caveats are documented in
the task README. The implementation does not claim a closed-form normalizer for
simple-path rejection or for the random hint-selection mechanism.

## Full-size AR launch

A full run was launched on 2026-09-19, after confirming the intended model was the
512-wide saved Sudoku AR architecture:

- W&B: https://wandb.ai/rz-text-dlp/hubpaths/runs/e7a712c284d4
- Run directory: `outputs/hubpaths_ar_language_like_20k_seed1_20260919/`
- Dataset: `data_cache/hubpaths/language_like_100k_seed0/`
- Model: 8 blocks, width 512, 8 heads; **35,448,083 parameters** with the
  10,003-token vocabulary. The same architecture has 25,207,308 parameters with
  Sudoku's 12-token vocabulary.
- Schedule: 20,000 optimizer steps, global batch 256, four accumulation steps,
  learning rate 3e-4, 2,500-step warmup; GPU 2, BF16 mixed precision.
- Data: 100,000 training examples and 512 each for validation, test, and OOD;
  three waypoints, no hints, sequence length 82. All 101,536 sequences passed
  batched verification, and every unordered endpoint-pair split intersection
  was empty. Coverage top-ups were disabled for this run.

The detached supervisor saves `run.json`, `status.json`, `train.log`, a frozen
source snapshot, and checkpoints. The launch configuration is
`configs/experiment/hubpaths_ar_sudoku_size.yaml`; the runner is
`new_scripts/launch_hubpaths_ar.py`. Live convergence should be assessed from
this run, not the 300-step smoke models described above.

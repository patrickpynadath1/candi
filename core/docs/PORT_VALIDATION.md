# Spherical and permuted autoregressive port validation

Both ports are self-contained within `core/`. RePlaid remains deferred.

## Spherical flow matching on Sudoku Hard

The validation run used `sudoku_spherical_truncated_adaptive`: the original
512-dimensional trainable unit embeddings, 512-wide eight-block SphereDiT,
batch size 256, learning rate 0.0003, 2500-step warmup, and EMA 0.9999.
The run was shortened to **2000 optimizer steps**; the experiment preset retains
the original 20000-step budget. This is an early-training check, still within
the original learning-rate warmup, not a completed-budget reproduction.

The named Sudoku preset preserves physical-coordinate adaptation. The generic
S-FLM adaptive schedule defaults to the original sampled-time (`legacy_time`)
coordinate. Truncation uses the original vocabulary/dimension bound. The schedule
completed 19 refits, with the last at step 1950 and a full 12800-observation buffer.

Fresh checkpoint loading, EMA evaluation, 180 model calls, and unchanged default
posterior shaping produced:

| Evaluation | Solved | Cell accuracy | Digit accuracy | Clue consistency |
| --- | ---: | ---: | ---: | ---: |
| Fixed 128-puzzle subset | 20/128 (15.625%) | 84.621% | 83.102% | 100% |
| Full 2000-puzzle validation split | 200/2000 (10.0%) | 84.338% | 82.791% | 99.35% |

Clue consistency measures the generated answer's agreement with puzzle clues;
the input prompt itself is preserved during sampling. Cell accuracy includes
the task's sequence formatting tokens; digit accuracy measures the grid digits.

Artifacts: `core/outputs/spherical_sudoku_hard_validation/`, including
`config.yaml`, `checkpoints/last.ckpt`, `evaluation_latest_sampler.json` (subset),
and `evaluation_full.json`. The subset evaluation took 10.3 seconds and the full
evaluation 142.3 seconds on an RTX A6000. Training and evaluation use float32
posterior normalization after the bfloat16 transformer, preserving the original
implementation. Geometry and sampling probability calculations use float64.

## Permuted autoregressive on Text8

No Sudoku training or evaluation was run for this method. The real Text8 corpus
uses its 90M/5M/5M character split and sequence length 256. The validation model
has 3.2M parameters (width 256, four blocks, eight heads), batch size 128,
learning rate 0.0003, 200-step warmup, and EMA 0.999.

Training ran for 3000 steps, then resumed the full Lightning checkpoint through
10000 steps. Optimizer, scheduler, EMA, and RNG state were restored; the
non-stateful data loader does not reproduce an uninterrupted example stream.
The named `permuted_autoregressive_text8` preset supplies the same hyperparameters
with a single 10000-step budget.

Generation used random permutations, KV caching, temperature 1, no top-k/top-p
filtering, and sampled token commitment. All 255 target positions were filled,
including positions following EOS in generation order. Metrics over 128 EMA
samples, using the training-split word vocabulary:

| Metric | Generated at 10000 steps | Held-out Text8 |
| --- | ---: | ---: |
| Valid-word rate | 68.20% | 99.01% |
| Valid-word rate, length >=3 | 60.03% | 98.80% |
| Valid-word rate, length >=5 | 29.90% | 97.97% |
| Valid-word rate, length >=7 | 9.07% | 96.92% |

The generated corpus contains 5500 words, 3751 valid words, 2995 distinct words,
and 1258 distinct valid words. The 3000-step valid-word rate was 61.69%.
The user accepted this as the initial port check; it is not a large-model quality
benchmark. All 128 unselected samples are retained for inspection.

Artifacts: `core/outputs/permuted_autoregressive_text8/`, including
`samples_3000.txt`, `samples_10000.txt`, `evaluation_10000/results.json`,
`checkpoints/last.ckpt`, and `run_notes.md` with the invocation. Final generation
took 22.7 seconds on an RTX A6000, with 255 model calls per sample.

## Implementation checks

- Full core suite: **245 passed, 8 skipped** in the sandbox; CUDA tests skipped
  there were checked separately with GPU access.
- Spherical focused CUDA suite: **9 passed**, including float32 posterior
  normalization during bfloat16 training, geometry, velocity, adaptation, and
  checkpoint tests.
- Permuted autoregressive focused CUDA suite: **17 passed**, including an
  independent attention reference, arbitrary-position cache equivalence,
  bfloat16 execution, causal visibility, ordering, and EOS behavior.
- Existing autoregressive/D3PM CUDA suite: **10 passed** with the extended causal
  model, protecting ordinary autoregressive caching.
- Changed Python files pass Ruff. The Text8 preset composes and instantiates
  the expected algorithm, vocabulary, sequence length, and cached sampler.

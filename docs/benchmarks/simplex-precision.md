# Opt-in simplex precision and memory controls

Baseline: Candi v2.1 `bb666ad12226b5e3fa2398bae327be995c424276`.
The supplied working tree was clean; no earlier FP32 implementation was present.
The original FP64 process and legacy projection remain the defaults.

Standalone Hydra overrides:

```text
process.precision=fp32 process.sampler_backend=triton process.chunk_size=128
model.projection_precision=bf16 model.projection_chunk_size=128
```

The same process keys live under a hybrid view's `process`, and projection
keys under its `representation`. Both `configs/hybrid_view/simplex.yaml` and
`configs/model/hybrid_uniform_simplex.yaml` expose them. Their representation
`state_precision` inherits the process precision so prompt clamping cannot
silently promote FP32 states to FP64. For programmatic construction, also pass
`state_precision="fp32"` to `SimplexRepresentation`.
Use `projection_precision=fp32` for the FP32 projection baseline. Sizes count
flattened batch/sequence rows, always retaining the **entire vocabulary**.
Zero disables chunking. Projection chunking requires explicit `fp32` or `bf16`.

## Implementation and limits

* The optional Triton kernel uses block-local Marsaglia–Tsang rejection and
  small-shape boosting, with FP32 transforms, acceptance calculations, and
  outputs. Zero shape returns negative infinity. Dirichlet normalization stays
  in log space; FP32 values are centered before reduction to avoid cancellation
  at tiny concentrations. Validation still synchronizes at the API boundary,
  but the rejection loop does not synchronize with the host.
* CUDA FP32 inputs without concentration gradients use Triton when requested.
  CPU, FP64, and autograd shape inputs use the original PyTorch sampler.
  Triton must be installed to request the CUDA fast path.
* Each fused call consumes a CUDA-generator seed tensor; there is no `.item()`.
  Resetting `torch.manual_seed` reproduces a fixed backend/shape/chunk layout.
  Successive calls draw fresh samples. Different chunk sizes or backends need
  not produce identical samples: validate their distributions, not equality.
* FP32/BF16 expectation projections compute exponentials in FP32 and preserve
  embedding-table gradients. Hybrid table normalization and scaling remain in
  place. BF16 GEMMs round operands and GEMM outputs; table-gradient contributions
  accumulate across chunks in FP32. CUDA GEMM accumulation behavior also depends
  on PyTorch's `allow_bf16_reduced_precision_reduction` setting. Benchmarks set
  it to false, and disable TF32 for the FP32 comparison. Library code does not
  modify global precision settings.
* Projection backward recomputes each row chunk's probabilities from the saved
  log state. It does not save a full additional probability/cast tensor per
  forward pass. Full state storage, a requested state gradient, the table, and
  table gradient are unavoidable here. There is also a table-sized temporary
  gradient contribution. Chunking does not make the state sparse.
* Corruption chunks limit temporary tensors in the ordinary non-differentiable
  sampling path. Autograd concentration inputs fall back to PyTorch and can
  retain rejection-loop graphs; this path does **not** promise bounded backward
  temporary memory. Reverse bridges use the selected precision/backend but are
  not chunked by this setting.
* On tested PyTorch 2.5.1, `torch.compile` supports the sampler with graph breaks
  around validation and the explicit eager Triton boundary. Full-graph sampler
  capture is not supported. The projection is tested compiled including backward.
* FP32 has finite dynamic range: extraordinarily small positive shapes whose
  log samples exceed FP32 range cannot be represented. Tests cover `1e-8` through
  `1000`; arbitrary subnormal concentrations are not guaranteed.

## Reproduction

Run from the repository root with exactly one idle A6000 exposed. These commands
never stop, restart, or alter training jobs. The tested UUID is recorded in JSON.

```bash
export CUDA_VISIBLE_DEVICES=GPU-707ec8ba-9bbb-f97c-7b07-6abe6335ec64
export OMP_NUM_THREADS=4
export PYTHONPATH=.
PY=/home/patrick/.codex/worktrees/22a3/duo/.venv/bin/python
$PY -m pytest core/tests/test_simplex_precision.py core/tests/test_simplex.py core/tests/test_adaptive_view_clock.py core/tests/test_hybrid_diffusion.py core/tests/test_hybrid_composition.py -q
$PY scripts/benchmark_simplex_precision.py
$PY scripts/validate_simplex_precision.py
```

The benchmark uses separate subprocesses, two warmups, three synchronized
steady-state measurements, peak allocated/reserved CUDA memory, and explicit
OOM/error records. Compilation is excluded through warmup. `projection_backward`
measures forward plus table-gradient backward; `combined` includes corruption,
projection, and table-gradient backward. Those timed paths do not request a state
or concentration gradient. Numerical tests separately exercise state gradients.
The embedding table is allocated in every timing worker, including corruption-only
workers, and is included in the reported memory. The full target log state alone
is 1.50 GiB in FP32 or 3.00 GiB in FP64.
Small workload: `(2, 32, 4097, 768)`; target: `(16, 512, 49153, 768)`.

Results live in `results/simplex_precision/benchmark.json` and
`results/simplex_precision/numerical-errors.json`; commands, versions, precision
settings, errors, and individual timing repetitions are retained.
These numerical and performance tests do not establish training-quality equivalence.

## Measured target results

75 successful cases and 3 OOMs across both workloads. Medians of three
measurements on one RTX A6000; allocated and reserved memory are GiB.

| Sampler | Rows/chunk | Corruption ms | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|
| fp64 | all | OOM | 43.642 | 45.193 |
| fp32 | all | 1126.82 | 27.516 | 28.699 |
| fused | all | 110.10 | 7.644 | 7.672 |
| fp64 | 128 | 3755.77 | 4.014 | 4.094 |
| fp32 | 128 | 1037.43 | 2.092 | 2.148 |
| fused | 128 | 127.59 | 1.782 | 1.818 |
| fp64 | 512 | 3903.25 | 6.633 | 6.822 |
| fp32 | 512 | 1053.67 | 3.446 | 3.541 |
| fused | 512 | 115.58 | 2.203 | 2.297 |

| Projection | Rows/chunk | Forward ms | Forward + backward ms | Backward peak allocated |
|---|---:|---:|---:|---:|
| fp32 | all | 31.38 | 58.91 | 3.485 |
| bf16 | all | 17.46 | 34.28 | 4.223 |
| fp32 | 128 | 39.98 | 115.18 | 1.985 |
| bf16 | 128 | 21.55 | 108.72 | 2.126 |
| fp32 | 512 | 37.57 | 76.56 | 2.057 |
| bf16 | 512 | 24.84 | 59.71 | 2.197 |

Fused FP32 corruption plus BF16 projection/backward took **142.19 ms**
unchunked (7.660 GiB allocated), **173.95 ms** at 512 rows (2.219 GiB),
and **236.48 ms** at 128 rows (2.126 GiB). Chunking saves temporary memory
but repeats GEMMs and table-gradient accumulation; it is not a universal speedup.
The unchunked FP32 PyTorch corruption plus BF16 projection/backward took
1160.24 ms. The unchunked FP64 corruption and both combined cases OOMed.

BF16 unchunked projection uses *more* peak allocated memory than FP32 here
because it holds FP32 probabilities plus BF16 casts. Row chunking bounds those
temporaries. Results are eager timings; compile compatibility is tested separately.

## Validation results

114 CUDA tests passed across the precision, existing simplex, adaptive-clock,
and hybrid suites. CPU precision/fallback tests: 29 passed, 2 CUDA-only tests
skipped. JUnit XML and text summaries are saved alongside the results.

Distribution checks cover Gamma and Dirichlet moments and log-Gamma moments
against digamma/trigamma theory from concentration `1e-8` to `1000`. Log-Gamma
checks are especially useful at tiny concentrations, where raw Gamma moments
converge slowly due to rare large draws. Monte Carlo checks use statistical
tolerances; they do not prove equality of distributions.

Maximum relative L2 errors across dense/sparse projection cases and tested chunks:

| Precision | Output | State gradient | Table gradient |
|---|---:|---:|---:|
| fp32 | 3.2389e-07 | 2.82973e-07 | 1.57633e-07 |
| bf16 | 0.00286214 | 0.00293595 | 0.0029097 |

Fused Dirichlet normalization max absolute error: 2.08616e-07.

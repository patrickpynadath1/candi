> Archived experiment: custom accumulator code was removed on 2026-11-01.
> Measurements are retained; accumulator configs are no longer runnable.
> Normal/block diffusion profiling remains supported.

# Context-memory preflight (2026-11-01)

Pilot training is paused pending a useful speed–memory tradeoff. No 2k-step
pilot jobs have been launched. All results here are synthetic training-memory
checks, not text-quality measurements.

Hardware: NVIDIA RTX A6000. Microbatch8, 8layers, width512, BF16, eager execution,
AdamW forward+backward+update; EMA and Lightning/data-loader overhead excluded.
TinyStories sequence1024, text8 sequence256, block32. Existing baseline has
layer activation checkpointing; blockwise accumulator has nested block/layer
checkpointing and full cross-block gradients. Linear/gated states FP32.

Files under `preflight/` record configs, exact peaks and seconds per microbatch.
`blockwise-linear-profiler.txt` contains instrumented CPU/CUDA operation counts;
profiler timings have overhead, so use uninstrumented JSONs for throughput.

The early files in /tmp/context-memory-profile-nested and the non-baseline
entries of /tmp/context-memory-profile-text8 and -stories are INVALID: a wiring
omission meant they constructed ordinary uncheckpointed block diffusion.
They are deliberately not included here. TrainingModule wiring and regression
assertions now ensure the selected executor is actually installed.

The final protocol/budget is not decided while profiling continues. The exported
cache configs are preparation artifacts, not running jobs. Inference budget is
per generated block, not whole sequence. See docs/guides/context-memory-ablation.md.

## First execution comparison

TinyStories-shaped synthetic sequence, microbatch8, measured optimizer steps:

| Execution | Peak allocated MB (decimal) | Seconds/microbatch |
| --- | ---: | ---: |
| Checkpointed block-diffusion baseline | 1294 | 0.184 |
| Linear memory, single-block/nested checkpoints | 830 | 5.20 |
| Linear memory, full layerwise batching | 1504 | 0.189 |
| Linear memory, groups of 2 blocks | 735 | 1.84 |
| Linear memory, groups of 4 blocks | 718 | 0.885 |
| Linear memory, groups of 8 blocks | 822 | 0.536 |

The phase trace attributes original runtime to1.35s forward,3.79s backward,
0.006s optimizer. It records tens of thousands of tiny operations. Instrumented
CUDA self time~0.51s versus CPU self time~8.62s indicates substantial dispatch,
checkpoint/autograd overhead; profiler overhead means those totals must not be
read as uninstrumented throughput. Grouped execution preserves full gradients.
These short measurements are preflight, not long-run throughput estimates.

Select block_diffusion.memory_execution=grouped and memory_group_size=8 for the
current candidate compromise. The default remains blockwise for compatibility;
no prepared pilot configs were silently switched or submitted. The measurements
above predate the optimized delta and RNN implementations below.

Grouped gated-linear8:856MB /0.873s per microbatch. Final combined regression
suite:128 tests passed, including kernels, full gradients, context visibility,
cache invalidation, Hydra assembly, grouped/layerwise parity, and pre-existing
block/sublayer/depth-mixing execution paths.

## Optimized delta and RNN

Delta now solves token corrections together using a unit-lower-triangular
system, then composes block transforms with an affine scan. The recurrent
reference remains selectable for correctness checks. This is parallel PyTorch
algebra, not a fused FLA kernel. RNN uses public `torch.nn.RNN` with cuDNN on
CUDA, preserving the tanh recurrence and one trainable bias. Both return
exclusive block boundaries and propagate full gradients through incoming and
outgoing state. Delta/RNN source presets now select groups of eight blocks.

Same synthetic BF16 microbatch8 protocol, two warmups and five measured steps:

| Task | Execution | Peak allocated MB | Seconds/microbatch |
| --- | --- | ---: | ---: |
| TinyStories | Delta chunkwise, group8 | 935 | 0.820 |
| TinyStories | Fused tanh RNN, group8 | 761 | 0.543 |
| text8 | Delta chunkwise, group8 | 782 | 0.230 |
| text8 | Fused tanh RNN, group8 | 641 | 0.150 |

Earlier blockwise token-loop TinyStories measurements were delta845MB/13.0s
and RNN565MB/8.06s. These improvements include both grouping and kernel changes;
they do not isolate one optimization. Both optimized TinyStories variants use
less memory than the checkpointed baseline1294MB/0.184s but remain slower.
For text8, both use MORE memory than baseline602MB/0.079s. This is not a
universal memory improvement, and changing the workload may change the tradeoff.

Reports: `preflight/optimized-delta-*.json`, `preflight/optimized-rnn.json`.
Delta reports predate adding the version/group-size fields to the profiler:
they used PyTorch2.5.1+cu121, CUDA12.1 and group8, like the RNN reports.
No training-quality results are implied; the pilot remains paused.

Validation: the combined CPU/CUDA regression run passed152 tests, covering
checkpoint replay, prefix visibility, nonzero initial states, full gradients,
BF16 and existing block/depth execution. An additional production-shaped delta
CUDA test passed separately in the17-test delta suite. Ruff passed. No test
tolerances were loosened to accommodate the backend changes.

FLA was assessed in an isolated `/tmp/candi-fla022-env`, not the shared training
environment. FLA0.2.2 and0.2.0 with Triton3.1 failed during import in Triton JIT
parsing; trying Triton3.2 with PyTorch2.5.1 instead broke Inductor's
AttrsDescriptor compatibility. The unverified FLA adapter was removed. No
optional package or shared-environment upgrade is required by the shipped code.

## Boundary-only recomputation investigation

The implementation already propagates future-loss gradients through boundary
states via autograd. We tested removing the inner layer checkpoints while
retaining group checkpoints, using `checkpoint_memory_layers: false`.
Execution counters confirm three evaluations per layer with nesting versus two
with boundary-only checkpoints. This counts forward evaluations, not total
training FLOPs; backward computation is still required in both cases.

Matched final-code measurements on GPU7, TinyStories-shaped microbatch8, group8,
BF16, two warmups/five measured steps, same seed and optimizer protocol:

| Memory | Nested time | Boundary-only time | Nested peak MB | Boundary-only peak MB |
| --- | ---: | ---: | ---: | ---: |
| Linear | 0.482s | 0.315s | 816 | 1958 |
| Delta | 0.824s | 0.604s | 935 | 2820 |
| RNN | 0.440s | 0.300s | 757 | 1738 |

Boundary-only is27–35% faster here, but consumes2.3–3.0 times as much peak
memory. All three exceed the earlier checkpointed baseline's1294MB. These are
short synthetic measurements, not sustained throughput; use the paired records
rather than mixing timings from earlier profiling sessions. Identical final
losses within each pair also support preserved execution/RNG behavior.

The compact boundary gradient does not replace the group's internal activations
needed to compute parameter gradients. Boundary-only replay reconstructs all
layers' saved tensors in a group; nested checkpointing keeps their peak lower
by replaying individual layers later. The repeated work has a memory purpose.

We also measured all four memories at groups2/4/8 on both tasks:
`preflight/boundary-only-{tinystories,text8}.json`. Smaller groups lower peak
memory but increase sequential overhead. For example, RNN TinyStories group2
uses845MB/1.168s versus group8's roughly1745MB/0.314s in that run. None of the
tested boundary-only text8 settings used less than the historical602MB baseline.
The initial size sweep ran on GPUs6/7 concurrently; the paired policy check used
the same GPU and final code for both policies.

Validation:166 regression tests passed, then9 CUDA tests for both checkpoint
policies and8 Hydra-wiring tests passed. Includes full cross-group gradients,
dropout RNG preservation and explicit replay counts. Ruff and diff checks passed.
The context-memory residual helper uses an equivalent eager expression because
TorchScript's differentiated helper failed saved-tensor replay with group-only
checkpoints. Defaults remain nested; pilot training remains paused. Finer-grained
selective checkpointing remains an unmeasured possible improvement.

### MLP-only checkpoint probe

A temporary `/tmp/probe_mlp_checkpoint.py` wrapper checkpointed only
PositionwiseMLP within each outer group, leaving attention/accumulator internals
uncheckpointed. Linear memory, TinyStories shape, group8, same synthetic protocol:
0.399s/1740MB. This is between boundary-only0.315s/1958MB and
fully nested0.482s/816MB, but still exceeds the checkpointed baseline1294MB.
Final synthetic loss matched the paired references; this probe has not received
full gradient-parity validation and was not integrated into production configs.
Record: `preflight/linear-mlp-only-prototype.json`. It rules out assuming that
checkpointing just the expanded MLP activations is sufficient at this setting.

## Fusion investigation

The optimized eager linear model still launched12335 GPU kernels in the profiled
step. aten::copy_, mul and add accounted for about37% of recorded self CUDA
time. The profiler adds overhead; those percentages are attribution, not
uninstrumented training throughput. See `preflight/fusion-operation-profile.txt`.

Saved-tensor storage attribution with checkpointing OFF found5397MB of distinct
nonparameter backing storage across the complete forward: accumulator reads1355MB,
scans403MB, MLPs1242MB, normalization1109MB, other block-functional storage944MB,
remaining345MB. Storage is attributed to its first saver, aliases deduplicated;
parameter casts may be included. This is NOT the checkpointed peak or exact
allocator liveness. It shows that accumulator pointwise intermediates and
normalization matter alongside the MLP. Raw data: `fusion-saved-storage.json`.

Compiled only the repeated `_parallel_layer` computation, keeping nested
checkpoints and FP32 accumulator states. An equally compiled ordinary transformer
layer provides the baseline. Initial10-step synthetic TinyStories measurements:

| Execution | Seconds/microbatch | Peak allocated MB |
| --- | ---: | ---: |
| Eager checkpointed baseline | .181 | 1294 |
| Eager linear group8 | .435 | 816 |
| Compiled checkpointed baseline | .139 | 1114 |
| Compiled linear group8 | .234 | 809 |
| Compiled linear group16 | .153 | 970 |
| Compiled linear group32 | .127 | 1255 |

Warmup/compilation is excluded from timing. Compile can change dropout random
streams, so synthetic loss values across compiled/eager runs are not parity
evidence. A separate two-layer, dropout-free, BF16 comparison had0.267% relative
parameter-gradient norm error and.00586 maximum logit difference; earlier clean
context still received gradients. This verifies that prototype, not downstream
quality or every accumulator. Raw records and resolved configs are under
`preflight/fusion-*`; no production configuration or pilot was switched.

The reusable profiler now accepts `--compile-target memory-layer` or
`--compile-target transformer-layer`, with `none` default. It compiles repeated
layers rather than the whole algorithm. Use the saved configs, microbatch8,
warmup3, and20 measured steps to reproduce the follow-up checks.
Fusion substantially reduces the slowdown. It does not yet establish a model
that beats the equally compiled baseline on both memory and time: group16
saves memory but is a little slower; group32 is faster but uses more memory.

Same-GPU20-step confirmation via the reusable profiler: compiled baseline
1117.5MB/.14036s; compiled linear group16 972.6MB/.15308s. User now requires
at least50% lower total peak GPU memory with no steady-state slowdown.
No tested candidate meets this. The usual FP32 Adam setup retains parameter,
gradient, first-moment and second-moment tensors:16bytes per parameter, about
408MB for25.5M parameters. Half the compiled TinyStories baseline leaves only
about150MB beyond this floor. For text8, half the earlier602MB baseline is below
its approximately405MB model/gradient/Adam floor. This floor assumes all tensors
are simultaneously GPU-resident, as in the current training step; alternatives
that stream gradients or change residency/precision require separate evaluation.

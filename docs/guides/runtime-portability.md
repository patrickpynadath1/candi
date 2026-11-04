# Runtime portability checks

Use the locked uv environment for comparisons. Changing Torch, GPU backend or
hardware is an experiment-environment change; record it with the results.

Full training checkpoints include optimizer and NumPy RNG state. The training
runner explicitly uses Lightning's `weights_only=False` resume option for these
trusted local artifacts. Do not use this path for untrusted downloaded pickles.
There is no need to set a process-wide Torch loading override.

The default sampler tests check repeatability within the runtime, prompt
preservation, call counts and metadata. Frozen reference-loop tests compare old
and new algorithms in the same runtime; distribution/moment tests cover the
continuous kernels. These checks do not prove training-quality equivalence.

Historical byte-level sampler snapshots remain unchanged. Run them explicitly:

```bash
CANDI_CHECK_LEGACY_HASHES=1 uv run --locked pytest core/tests/test_sampler_equivalence.py -k legacy
```

These hashes include every state byte and the CPU RNG state. Torch multinomial
streams and floating-point kernels can change across versions/platforms.
The original fixture did not record a complete hardware/runtime fingerprint:
even Torch 2.5.1 alone is not sufficient to guarantee matching hashes.
Do not regenerate the fixture merely to make a new platform pass. In particular,
the reported simplex trace differences on ROCm still need per-step numerical
investigation; passing portable checks is not a resolution of their cause.

Tests that fake GPU visibility or subprocess discovery must run in a child
process. CUDA and Triton cache driver discovery, so restoring environment
variables alone does not undo a poisoned cache.

On a new ROCm deployment, rerun the suite, then compiled simplex tests in both
isolation and after the programmatic sweep tests. Finally run a compiled Ray
training/resume/generation smoke at the actual production batch size. CPU tests
cannot validate GPU compilation or multi-node Slurm startup.

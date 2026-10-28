# BlockGen fixed-block recipes

[BlockGen: Flexible Blockwise Sequence Modeling with Hybrid Samplers](https://arxiv.org/abs/2606.02241)
has both fixed-block baselines and models trained on mixtures of block sizes.
The fixed-block presets and optional training mixtures reuse the existing
block modifier, losses and ancestral samplers; no separate model or loop was
added. Reference code: [jdeschena/blockgen at 6f52b0d](https://github.com/jdeschena/blockgen/tree/6f52b0da3aa961d7cd268fe2d7c20cd94b3f51ae).

| Experiment | Information path | Training objective | Inference |
| --- | --- | --- | --- |
| `blockgen_masked` | Absorbing mask within each block | Clean-token CE with observed-token preservation | Masked ancestral, sequential blocks |
| `blockgen_uniform` | Uniform replacements within each block | Plug-in uniform ELBO | Uniform ancestral, sequential blocks |

The uniform preset is deliberately **not** posterior CE. The masked preset uses
CE during both training and validation; the reference switches validation to the
masked ELBO. Consequently, do not report its `val/nll` as a paper-matched ELBO.
Changing `objective=weighted_ce` changes both training and validation rather than
reproducing that switch.

Both use block size 16 by default, token-causal clean-stream attention
(`x0_causal=true`), clean-prefix KV caching and no auxiliary clean-stream AR loss.
This differs from the reference's fixed-block scripts, which use block-causal
clean-stream attention. The conditioned
DiT has 12 layers, width 768 and 12 heads, but diffusion time conditioning is
turned off, matching the reference setting. Dataset, batching and training budget
remain the shared local defaults; these presets do not silently select TinyGSM,
OpenWebText, eight GPUs, or million-step training.

```bash
# Inspect without launching training.
uv run --locked python main.py +experiment=blockgen_masked task=text8 --cfg job
uv run --locked python main.py +experiment=blockgen_uniform task=tinystories --cfg job

# Change block size independently of task and corruption.
uv run --locked python main.py +experiment=blockgen_masked task=text8 \
  block_diffusion.block_size=32 trainer.max_steps=20000
```

The default sampler performs 15 reverse updates plus terminal decoding:
**16 NFE per generated block**, not per sequence. Sequence NFE sums the calls
across generated blocks. Prompt-only blocks incur no calls. Change sampling
budget and temperature through `sampler.steps` and `parameterization.temperature`.

The existing `tinygsm_blockgen_masked` and `tinygsm_blockgen_uniform` presets
remain available for the separate TinyGSM data protocol and reference training
budget. They are also fixed-block variants; their existing sampler budget is
unchanged by these task-independent presets.

## Mixture training

Use `+experiment=blockgen_masked_mixture` or
`+experiment=blockgen_uniform_mixture` for a shared model trained with block
sizes 1 and 16, with probabilities 0.05 and 0.95. The singleton branch uses exact
pure-noise corruption and CE. Larger blocks inherit the base recipe's objective
and time sampler. Uniform therefore retains its plug-in ELBO at size 16.

```yaml
block_diffusion:
  block_size: 16  # generation and fixed-size validation default
  x0_causal: true
  training:
    _target_: core.modifiers.block_diffusion.mixture.BlockSizeMixture
    sizes: [1, 16]
    probabilities: [0.05, 0.95]
    sampling: stratified  # or random
    seed: ${seed}
    overrides:
      1:
        pure_noise: true
        time_sampler:
          _target_: core.time_sampling.fixed.FixedTimeSampler
          time: 1.0
        objective:
          _target_: core.objectives.cross_entropy.CrossEntropy
```

One size is selected per GPU microbatch. Stratification shares a deterministic
random offset and permutation across ranks and accumulation slots for each
optimizer step. Sampling probabilities already supply the mixture weights; the
loss is not multiplied by those probabilities again. A singleton size list uses
the same path without consuming a block-selection random draw.

The block modifier scopes the selected size to the current forward call; it does
not change the saved inference size or swap model/objective modules. Overrides
are explicit per-call dependencies and are checked against the algorithm's loss
interface at construction. The size sampler itself is independent of corruption:
continuous recipes can use it without singleton overrides, or supply their own
supported time sampler and objective. Exact `pure_noise` overrides currently
support the masked and uniform token paths with CE. `FixedTimeSampler(time=1)`
alone means the schedule endpoint, which may retain signal.

Current mixture training requires fixed-length sequences; insertion/deletion
combinations fail at construction. Validation uses the configured fixed block
size and base objective. It does **not** report mixture likelihood. Training
mixtures do not automatically change generation block sizes or sampler tuning.

## Full BlockGen components not yet implemented

The mixture-model recipe still needs:
- Mixture likelihood evaluation.
- AR-informed predictor-corrector (ARPC) or entropy-informed corrector (EIPC)
  transitions, including their step schedules and additional NFE accounting.

`block_diffusion.clean_loss=autoregressive` adds an auxiliary loss; it is not a
substitute for singleton pure-noise training. Likewise, entropy-bounded
unmasking is not EIPC. The new YAMLs explicitly document these limits rather
than labelling those existing options as implementations of the full method.

The focused tests cover composition on Sudoku/text8/TinyStories, finite loss and
gradients, prompt preservation, vocabulary support, cached/uncached generation
parity, and per-block NFE at two block sizes. No BlockGen training job is added
to the feedback/depth ablation automatically.

# BlockGen fixed-block recipes

[BlockGen: Flexible Blockwise Sequence Modeling with Hybrid Samplers](https://arxiv.org/abs/2606.02241)
has both fixed-block baselines and models trained on mixtures of block sizes.
These presets implement the **fixed-block baselines**. They reuse the existing
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

Both use block size 16 by default, block-causal clean-stream attention
(`x0_causal=false`), clean-prefix KV caching and no auxiliary clean-stream AR loss.
The reference uses this attention choice for fixed-block models. The conditioned
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

## Full BlockGen components not yet implemented

The main mixture-model recipe additionally needs:

- Weighted block-size sampling, including distributed/accumulation stratification.
- Pure-noise singleton blocks for the AR component and size-dependent objectives.
- Mixture likelihood evaluation and causal clean-stream attention for that model.
- AR-informed predictor-corrector (ARPC) or entropy-informed corrector (EIPC)
  transitions, including their step schedules and additional NFE accounting.

`block_diffusion.clean_loss=autoregressive` adds an auxiliary loss; it is not a
substitute for the paper's block-size mixture. Likewise, entropy-bounded
unmasking is not EIPC. The new YAMLs explicitly document these limits rather
than labelling those existing options as implementations of the full method.

The focused tests cover composition on Sudoku/text8/TinyStories, finite loss and
gradients, prompt preservation, vocabulary support, cached/uncached generation
parity, and per-block NFE at two block sizes. No BlockGen training job is added
to the feedback/depth ablation automatically.

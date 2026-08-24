# Rolling block diffusion

The block-diffusion implementation doubles the input into noisy and clean
streams, uses a block-causal attention mask, and applies anchored tokenwise
AdaLN conditioning. At noise level zero the modulation is exactly zero, so the
model reduces to the ordinary causal AR network.

## Noise-level modes

`algo.noise_level_mode` controls how training noise levels are sampled:

- `monotone`: sample nondecreasing position-dependent levels independently
  inside each block.
- `sequence_constant`: sample one level per training sequence and use it for
  every noisy position.

The sequence-constant schedule is the simplest recommended starting point for
scaling experiments such as OpenWebText. It removes positional schedule
complexity while retaining the doubled-stream attention structure, clean AR
teacher path, embedding-level flow corruption, and rolling inference API.
Preliminary 10k-step Text8 runs learned speculative acceptance faster with the
constant schedule than with the matched monotone schedule; the final controlled
ablation should still be consulted before treating this as a quality result.

Example overrides:

```bash
algo=block_diffusion \
algo.block_size=16 \
algo.noise_level_mode=sequence_constant \
algo.noise_curriculum_start_step=2500 \
algo.noise_curriculum_end_step=5000 \
algo.clean_loss_weight=0.25 \
algo.logit_kl_weight=0
```

For a longer OWT run, scale the curriculum boundaries with the total training
budget rather than copying the Text8 step counts literally. The delayed
curriculum remains exactly AR before `noise_curriculum_start_step`, ramps
linearly to full noise at `noise_curriculum_end_step`, and stays at full noise
for the remainder of training.

The ready-to-run launcher is
`scripts/train_owt_block_constant_noise.sh`. Its defaults use a one-million-step
budget with an AR phase through 250k and a linear noise ramp from 250k to 500k.
All scale, batch, validation, and curriculum values are environment variables;
see the header of the script for an invocation example.

## Training-time rolling evaluation

`Text8SamplingCallback` can run after every validation event. Its generation is
isolated with `torch.random.fork_rng`, so evaluation does not alter subsequent
training corruption or dropout randomness. For speculative block decoding it
logs:

- proposal-token acceptance rate;
- mean accepted prefix length per verification window;
- mean verified advance, including a correction token on rejection;
- mean NFE, token entropy, and decoded examples.

The accepted-prefix metric is bounded by the block size and includes fully
accepted windows. This is preferable to `accepted / rejected`, which is
unbounded and overstates progress when rejections are rare.

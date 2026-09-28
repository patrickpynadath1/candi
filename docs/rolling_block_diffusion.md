# Rolling block diffusion

## CANDI Gaussian embedding schedules

`algo=block_diffusion_embedding` adds CANDI's Gaussian embedding schedule
mappings to the existing rolling model. It retains the shared causal
transformer, clean/noisy streams, and `noise_level_mode=monotone`.

| Variant | Overrides |
| --- | --- |
| Linear interpolation | `algo.interpolation_schedule=linear noise.adaptive=false` |
| Linear rank | `algo.interpolation_schedule=linear_rank noise.adaptive=false` |
| Geometric rank | `algo.interpolation_schedule=geometric_rank noise.adaptive=false` |
| Adaptive linear rank | `algo.interpolation_schedule=linear_rank noise.adaptive=true` |

Adaptive timing also works with linear interpolation and geometric rank.
Use `noise=rank-adaptive` (or `rank-adaptive-cdcd` for the existing CDCD
estimator), with untruncated endpoints and `noise.eps=0`.

For each sampled position clock `t`, the model computes progress
`rho = 1 - noise.alpha_t(t)` and maps that progress to an interpolation
coefficient `u`. Corruption is `(1-u)*embedding + u*Gaussian_noise`.
The same monotone mapping is applied in training, denoiser conditioning,
rolling warmup, refinement, and Euler/renoise transport. Position order is
preserved within each block even after an adaptive clock refit. Adaptation
uses valid positions' raw noisy next-token CE and their individual times;
the clean-stream auxiliary loss is excluded. This is the same estimator
machinery as CANDI, applied to the blockwise position-time distribution.

Rank mappings reuse `SphereRankSchedule` with default bounds 0.01–0.49.
Unlike the finite-endpoint CANDI experiments, rolling requires an exactly
clean anchor and a pure Gaussian prior. `rank_endpoint_width=0.01` therefore
adds linear transitions over the first/last 1% of corruption progress; rank
coupling applies over the interior. Zero endpoint width is rejected for
rank variants. Rank calibration requires unit embeddings
(`embedding_scale=1`). SLERP and one-hot rank formulas are different
corruption geometries and are not options of this Gaussian embedding variant.

TinyGSM example (inside a suitable GPU allocation, with an installed uv env):

```bash
uv run --no-sync python main.py \
  +experiment=tinygsm_block_embedding \
  algo.interpolation_schedule=linear_rank noise.adaptive=true \
  algo.noise_level_mode=monotone \
  algo.noise_curriculum_start_step=2500 \
  algo.noise_curriculum_end_step=5000 trainer.max_steps=50000 \
  sampling.predictor=block_diffusion_euler
```

`tinygsm_block_embedding` selects the appropriate algorithm, rank-adaptive
noise configuration (adaptation disabled by default), and verifier-free
rolling Euler sampler. Set `sampling.predictor=block_diffusion` for the
speculative sampler. Both samplers support all the mappings above. Existing
`algo=block_diffusion` behavior and checkpoint structure are unchanged.

### TinyGSM block-64 Slurm training

From the repository root, submit `job_block_embed_linear.sh` and
`job_block_embed_linear_rank.sh`. Both use monotone blocks of 64, fixed
non-adaptive schedules, 512-token sequences, global batch 512, 250,000
optimizer steps, and the existing small model. The clean AR curriculum lasts
2,500 steps, then ramps to full noise at step 5,000. Evaluation uses the
verifier-free rolling Euler sampler.

`job_block_embed_linear_no_noise_warmup.sh` adds a separate linear-interpolation
run with both noise curriculum boundaries set to zero: the full noise range
is available from the first step. It retains the 2,500-step learning-rate
warmup and uses a separate run name and checkpoint directory.

These request eight H100s for 48 hours on normal QoS with requeue enabled.
They use the installed uv environment and existing TinyGSM cache, and log to
`candi-tinygsm`, group `block64-embedding-schedules`. Each job gets its own
checkpoint directory and stable W&B ID; requeued jobs select their newest
readable saved checkpoint. Both configs have CPU block-64 backward/sampling
checks; full-scale GPU training remains to be validated by the jobs.

## Original rolling model

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

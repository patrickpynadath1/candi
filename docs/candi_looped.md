# Looped CANDI

Select `algo=candi_looped sampling.predictor=candi_looped`, or use the
Sudoku preset:

```bash
.venv/bin/python main.py +experiment=sudoku_candi_looped
```

The backbone reuses CANDI's clean anchor/noisy token inputs, preconditioning,
time conditioning, and transformer blocks. Before the first block it adds
`W_feedback(LayerNorm(previous_hidden))` to the projected token input.
`previous_hidden` is the position-aligned output of the last transformer block,
before the vocabulary head. This adds a feature stream, not sequence positions.
The first pass uses zeros; the bias-free feedback projection makes this exactly
the ordinary CANDI input. The feedback projection has its own learned weights;
all transformer and vocabulary-head weights are shared across passes.

## Training

1. Sample CANDI's corruption once and run the entire batch with zero feedback.
2. Uniformly choose half the examples without replacement.
3. Recompute only those examples with their detached first-pass top hidden
   states. Keep tokens, noise, time, and anchor masks identical to the first pass.
4. Use the second prediction for selected examples and the first for the rest.
   Each example contributes exactly one token loss with the existing weighting,
   prompt masking, and adaptive-schedule statistics.

For odd batches, randomly round the subset size up or down with equal
probability. A singleton therefore receives a second pass with probability 0.5.
The subset is selected per local microbatch. First-pass predictions for selected
examples have no direct loss and receive no gradient through the detached
feedback. The first full-batch graph is retained for unselected examples, so
this implementation should not be assumed to have the same memory or backward
cost as a no-grad preliminary pass. Forward work is 1.5 batch-equivalents for
even batches; no training-speed or quality improvement is claimed without a run.

## Evaluation

Generation runs exactly one full backbone pass per diffusion timestep. The
first step uses zero feedback; each later step receives the detached top hidden
state from the immediately preceding step. The carry is local to one sampling
request and resets for each new batch. The `candi_looped` sampler uses CANDI's
standard transitions and reports `nfe = diffusion_steps`.

Standalone validation loss has no preceding diffusion timestep, so it retains
the two-pass self-conditioned loss evaluation with `algo.validation_loops=2`.
This setting does not change generation's one-pass-per-step behavior. The old
`algo.inference_loops` setting has been removed; existing model weights remain
compatible. Training's half-batch recomputation is unchanged.

The new algorithm and backbone have separate registry/config names. Existing
CANDI checkpoints lack the feedback projection and are not strict-load-compatible
with this variant. No checkpoint conversion or pretrained weights are included.

For a small recurrent-state bottleneck, see [register recurrence](candi_register_looped.md).
It carries only dedicated register positions and shares this training and
sampling procedure.

For either variant, use `sampling.predictor=candi_looped_entropy_bounded`
with `sampling.entropy_bound=0.1 sampling.greedy_clean_estimate=true` for
EB + greedy. The loop-aware sampler reports total and per-example NFE in full
backbone passes. `new_scripts/eval_sudoku_looped_eb_greedy.sh` evaluates a
checkpoint on hard Sudoku with EMA weights, previous-step feedback, and a
180-step limit.

For standard scheduled reveals with greedy token commits (no EB), select
`sampling.predictor=candi_looped sampling.greedy_clean_estimate=true`.
This retains stochastic reveal timing and the normal continuous transition,
but newly revealed tokens use the predicted argmax. Sampling still carries
the previous step's state with one backbone call per timestep.

## Related Work

- [Analog Bits (Chen et al., 2022)](https://arxiv.org/abs/2208.04202),
  Section 2 and Algorithm 1: zero conditioning with 50% probability, otherwise
  conditioning on a detached preliminary prediction. It feeds back a predicted
  clean sample; this variant instead uses the top hidden representation and
  selects an exact half-batch for even batches. Its sampler carries predictions
  across timesteps; our sampler similarly carries state across timesteps, but
  uses hidden features rather than predicted clean samples.
- [Self-conditioned Embedding Diffusion (Strudel et al., 2022)](https://arxiv.org/abs/2211.04236):
  continuous text diffusion in token embedding space with self-conditioning.
- [Scaling up Test-Time Compute with Latent Reasoning (Geiping et al., 2025)](https://arxiv.org/abs/2502.05171):
  recurrent depth and latent-state iteration for additional inference computation.
- [Reasoning with Latent Thoughts (Saunshi et al., 2025)](https://arxiv.org/abs/2502.17416):
  shared transformer layers iterated for latent reasoning, with theoretical and
  empirical comparisons against deeper models.
- [Looped Diffusion Language Models (Lee et al., 2026)](https://arxiv.org/abs/2605.26106):
  LoopMDM repeats a shared early-middle block inside a masked denoiser while
  applying the head and tail once. This variant instead reruns the full backbone
  with the same corrupted input and an additional top-state feedback stream.

These works motivate the experiment; this implementation is not a reproduction
of any one of them.

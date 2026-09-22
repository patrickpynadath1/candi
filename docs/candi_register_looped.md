# Register-Looped CANDI

Select `algo=candi_register_looped sampling.predictor=candi_looped`, or run:

```bash
.venv/bin/python main.py +experiment=sudoku_candi_register_looped
```

This alternative to [full-state looping](candi_looped.md) appends
`algo.num_registers=4` learned positions to the projected anchor/noisy token
sequence. They participate in the same bidirectional attention and time
conditioning as data tokens. Rotary positions follow the data positions.
Registers are internal hidden vectors: they have no vocabulary token IDs,
corruption, anchor masks, or direct token loss.

On the first pass, each register starts at its learned embedding. Subsequent
passes initialize each register with its learned embedding plus
`W_feedback(LayerNorm(previous_register_hidden))`. Only the final transformer
states at register positions are passed back, with shape `[batch, registers,
hidden_size]`. Data-position hidden states are discarded. Vocabulary logits
are returned only for the original data positions. The shared output head also
computes register logits internally, which are discarded.

Training inherits the full-state variant's procedure: corrupt once, run the
whole batch, choose exactly half uniformly without replacement for even batch
sizes (stochastic rounding for odd sizes), then recompute selected examples
with detached register states and identical corruption. Only each example's
final prediction receives the token loss. The transformer weights are shared
across passes. Registers learn through their attention contribution to token
predictions; gradients do not cross the feedback boundary.

Generation runs one full pass per diffusion timestep. The first step starts
with learned register embeddings; later steps add feedback from the previous
timestep's top register states. Register carry resets between sampling requests,
not between timesteps. Both `candi_looped` and `candi_looped_entropy_bounded`
report one NFE per executed timestep. Standalone validation loss still uses two
passes (`algo.validation_loops=2`), because it has no preceding sampling step.
Training's half-batch recomputation is unchanged.

The recurrent state is `R * D` values per example, versus `L * D` in the
full-state variant; attention processes `L + R` positions on every pass. This
reduces the size of feedback, not necessarily training peak memory, since the
full-batch graph is retained. Existing CANDI checkpoints are not strict-load
compatible. The backbone retains CANDI's zero-initialized AdaLN gates, so
register-to-token influence is initially zero and develops as gates learn.
Quality and convergence still need training experiments.

## Related Work

- [Recurrent Memory Transformer (Bulatov et al., 2022)](https://arxiv.org/abs/2207.06881)
  introduces special memory tokens that carry state between sequence segments.
  It is a close architectural precedent for a small recurrent token state.
  Here recurrence revisits the same corrupted sequence instead of moving to
  the next segment, and the feedback is detached.
- [Vision Transformers Need Registers (Darcet et al., 2023/ICLR 2024)](https://arxiv.org/abs/2309.16588)
  adds dedicated input tokens for internal computation. That work motivates
  explicit workspace positions but does not itself implement this iterative
  denoiser recurrence.
- [Analog Bits (Chen et al., 2022)](https://arxiv.org/abs/2208.04202)
  motivates probabilistic diffusion self-conditioning; our recurrent state
  consists of register features rather than a clean-sample prediction.

This combines architectural ideas from these works and is not a reproduction
of their training methods or reported results.

# Contextual latent diffusion with a clean token decoder

Use `+experiment=latent_autoencoder task=battery/text8` (or
`task=battery/tinystories`). This recipe is implemented but has not been queued
for a training campaign. It uses cached fixed-length block generation.

The encoder, normalization, latent denoiser and token decoder are separate model
components in `configs/model/latent_autoencoder.yaml`. Encoder, denoiser and token
decoder depths are independently configured under `latent`; defaults are8 each,
width512, block32. Both prediction streams reuse `EncoderDecoderModel` with zero
*additional* encoder layers: their context is already encoded by the shared clean
encoder. Thus they attend to final clean context representations, not the recent
encoder-K/V depth-selection variant.

For clean tokens x, compute z=N(E(x)). N applies per-position LayerNorm without
learned affine parameters, then L2 normalization to radius sqrt(hidden_size).
The Gaussian process and linear interpolation schedule are existing components:
z_t=(1-t)z+t*epsilon. The denoiser predicts z directly, not tokens or logits.

The recipe composes two independently configurable loss modules:

- `objective: latent_mse` averages squared error across latent dimensions.
- `algorithm.clean_loss: CrossEntropy` trains clean latents -> token decoder -> x.

Both have scalar weights (default1). No decoded-denoiser CE is included in the
training objective. `algorithm.detach_encoder_for_denoising: true` detaches the
latent target, noisy source and clean attention context used by the denoiser. The
encoder and token decoder then learn through clean decoding only. Setting it to
false allows latent MSE to update the encoder through all three routes. The clean
loss encourages information retention; neither it nor normalization guarantees
absence of collapse or that the decoder uses every latent dimension.

A separate no-grad decode of the denoiser prediction supplies token CE to the
existing CDCD time sampler. This is deliberately not latent MSE. The CDCD fitter
uses its existing post-optimizer-step update and checkpointed buffers. Training
logs `latent_mse`, `clean_decoding_loss` and `decoded_denoising_ce`, plus total loss.
Validation monitors `val/latent_objective` (the combined objective, not token NLL).

Visibility: the encoder is bidirectional within each block and causal across
blocks. Each noisy/decoded block can read only clean **past** blocks via context
attention. Its own clean representation reaches the denoiser only through
corruption. Because clean encoding mixes tokens within blocks, partially revealed
blocks are rejected. Generation accepts contiguous prompts ending on a block
boundary, or entirely given sequences. Arbitrary Sudoku clues are not supported
by this initial recipe. Fixed-length text8/TinyStories block prompts are supported.

`LatentAutoencoderSampler` uses the existing Gaussian reverse vector field and
fixed-grid loop. It starts each block from Gaussian noise, predicts clean latent
states along a linear time grid, then decodes the final predicted latent to tokens.
The training CDCD proposal does not automatically change this inference grid.
Choose `sampler.greedy` or `sampler.temperature` for terminal token decoding.
`sampler.steps` is **per generated block**; reported NFE sums denoiser updates,
one terminal denoiser call, and one token-decoder call per block. Clean context
encoding is not included in NFE. Encoder K/V and both context-attention streams
cache completed blocks across generation. Batch generation currently processes
examples serially for simple, correct handling of differing prompt lengths.

The recipe plugs into TrainingModule, its optimizers/EMA/checkpoints and the
standard task generation evaluator. It is not yet included in a tuning manifest;
any future sweep must account for its per-block budget and separate token decoder.
Current tiny synthetic tests check semantics and execution, not quality or
anti-collapse behavior. Tests cover loss decomposition, gradient ownership,
CDCD feedback, normalization, no future-context leakage, incremental caching,
prompt validation, sampling NFE, and checkpoint round trips.

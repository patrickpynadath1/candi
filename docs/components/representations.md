# Representations

Recipe context: [information path](../concepts/information-path.md).

A representation maps discrete tokens to continuous states and interprets a
clean-token posterior in that space. It owns the token embedding table when one
exists; the model's input projections are separate learned layers.

The boundary is deliberately tokenwise:

- `encode(tokens)` maps `[B,L]` token IDs to `[B,L,D]` states.
- `posterior_mean(probabilities)` maps `[B,L,V]` clean-token probabilities to an
  expected clean state.
- `decode(state)` returns `[B,L]` token IDs.
- `spec` records kind, vocabulary size, dimension, and unit-normalization guarantees.

## Implementations

| `representation` | Encoding / posterior mean / decoding |
| --- | --- |
| `one_hot` | One-hot / probability vector / coordinate argmax; `D=V` |
| `embedding` | Lookup in `E` / `p @ E` / argmax of `state @ E.T` |
| `centered_embedding` | Learned centered embedding geometry used by blurring, with post-update projection |

[EmbeddingRepresentation](../../core/representations/embedding.py) supports
`dimension`, `unit_normalized`, `trainable`, and `project_after_update`. A fixed
table is a registered buffer; a trainable table is a parameter. Unit normalization
is enforced whenever its codebook is used. Post-update projection requires both
trainability and unit normalization. Dot-product decoding is also nearest-neighbor
decoding when all rows have unit norm.

[CenteredEmbeddingRepresentation](../../core/representations/centered_embedding.py)
initializes feature columns according to blur rates and alternates centering and
row normalization during projection. Its codebook normalizes rows at use time;
it is a specific blurring geometry, not an alias for the generic embedding table.
The training lifecycle calls supported `post_optimizer_step` hooks.

The expectation of unit embeddings need not have unit norm. In particular,
self-conditioning on a spherical posterior mean retains its uncertainty-bearing
norm; do not renormalize it just because the individual rows lie on a sphere.

## Compatibility

[RepresentationRequirements](../../core/representations/spec.py) lets schedules
declare supported kinds, dimensions, vocabulary sizes, and normalization. Gaussian
construction checks these requirements and model dimensions. One-hot rank and
argmax-error calibration require one-hot geometry. Embedding sphere rank requires
unit embeddings of the specified dimension and uses an approximation based on
random sphere geometry. A common scale wrapper preserves these requirements.

Adding a tokenwise encoder/decoder requires implementing this boundary and
registering its state through the algorithm. A contextual encoder or one that
compresses sequence length needs additional posterior-mean and prompt-alignment
semantics; merely adding `encode` and `decode` is insufficient. See
[Gaussian diffusion](../methods/gaussian-diffusion.md) for recipe-level choices.

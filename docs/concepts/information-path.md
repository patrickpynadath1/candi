# Information path

The information path defines how tokens are represented and how their information
is degraded as time advances. It combines three choices:

| Component | Meaning | Examples |
| --- | --- | --- |
| [Representation](../components/representations.md) | Space in which corruption acts | Discrete tokens, one-hot vectors, embeddings, simplex states |
| [Corruption process](../components/processes.md) | Distribution or transformation of a clean state at a given noise level | Masking, uniform replacement, Gaussian noise, Dirichlet corruption, spherical paths, spectral blurring |
| [Noise schedule](../components/noise-schedules.md) | Time-to-corruption coefficients | Keep rate, Gaussian signal/noise scales, concentration, blur coefficients |

Representation is distinct from a model's input projection: the former determines
the corruption state space; the latter maps that state to hidden features.
Processes also supply transition mathematics used by inference update rules.

## Time warping and adaptation

A time warp changes how quickly a base path is traversed. In a hybrid view,
`time_map` maps the global clock to the local schedule's clock. A fixed analytical
warp and a loss-adaptive clock are different choices.

[`configs/information_path/time_map/cdcd.yaml`](../../configs/information_path/time_map/cdcd.yaml) configures a
loss-adaptive view clock for Gaussian, simplex, spherical, and blurring recipes.
It observes detached, masked per-example training cross-entropy, fits a monotone
mapping, and saves its state in the model checkpoint. Validation does not refit
it. The same fitted map is used during generation. The Gaussian-specific adaptive
schedule wrapper remains available for standalone Gaussian recipes; do not apply
both wrappers unintentionally.

An adaptive clock changes what each time means: the noise level encountered at
each training time and the physical spacing of inference points. A
[training time sampler](../components/time-sampling.md) instead changes how often
training visits each time and plays no role at inference. In training only their
combination and the loss weighting matter, and the two interact: an adaptive
schedule fits from the times the sampler draws. Training therefore passes each
draw's density to the schedule, and the schedule divides it out, so the schedule
it learns does not depend on the sampler (see the [recipe](recipe.md#noise-schedule-and-training-time-sampler)).

## Multiple paths and changing sequence length

[Hybrid views](../methods/hybrids.md) each own a representation, process, schedule,
local clock, and compatible update rule. A shared model combines their inputs.

[Insertion/deletion](../modifiers/variable-length.md) additionally changes which
positions exist. The birth distribution must match the process. In particular,
masked insertion starts from masks and uniform insertion starts from uniform
noise. Continuous birth rules may require predicted clean tokens before applying
forward corruption. These constraints are explicit even when other components
are shared.

See [the recipe overview](recipe.md) for the other two parts and
[component contracts](component-contracts.md) for compatibility requirements.

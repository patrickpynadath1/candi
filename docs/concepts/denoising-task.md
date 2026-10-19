# Denoising task

The denoising task specifies what the network must infer from the information path
and how training rewards that prediction.

| Component | Responsibility |
| --- | --- |
| [Backbone and conditioning](../components/models.md) | Transform noisy representations, prompts, time, and optional feedback into outputs |
| [Output parameterization](../components/predictors.md) | Give those outputs mathematical meaning, such as clean-token posterior or absorbing score |
| [Objective](../components/objectives.md) | Define the prediction loss, weighting, and auxiliary terms |
| [Training time sampler](../components/time-sampling.md) | Choose the time points presented during training |

A shared tensor shape does not imply a shared prediction meaning. For example,
uniform posterior cross-entropy and uniform plug-in formulations require their own
compatible objectives and reverse transitions. Temperature and top-k/top-p shaping
are inference settings; they do not alter the unshaped training prediction.

The algorithm coordinates these components. `TrainingModule` owns optimizer,
precision, accumulation, EMA, and checkpoint lifecycle. Learning rate and training
budget are experiment settings, not a new corruption process.

## Training modifiers

[Self-conditioning](../modifiers/self-conditioning.md) adds a feedback input and a
policy for obtaining it during training. Hidden-state and posterior-mean feedback
are different recipes. Their inference feedback policies must also be specified.

[Insertion](../modifiers/variable-length.md) adds count supervision and optionally
gap-content supervision. For discrete insertion, predicting gap content can be an
auxiliary task even when generated births remain pure noise. Continuous births
that need a clean-token estimate cannot simply discard that predictor.

[Adaptive information paths](information-path.md) receive detached loss feedback
from training coordination. Their fitted state belongs to the checkpoint; the
loss module does not need to own the schedule.

See [training](../guides/training.md) for execution and
[nested tuning](../guides/nested-tuning.md) for validation-based selection of
training hyperparameters.

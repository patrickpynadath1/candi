"""State-specific constraints, independent of posterior inference shaping."""

from .utils import exclude_token, fix_observed_tokens


class UnconstrainedPosterior:
    def exclude(self, logits):
        return logits

    def condition(self, log_probs, observed):
        return log_probs


class AbsorbingConstraints:
    def __init__(self, mask_index):
        self.mask_index = mask_index

    def exclude(self, logits):
        return exclude_token(logits, self.mask_index)

    def condition(self, log_probs, observed):
        return fix_observed_tokens(log_probs, observed, self.mask_index)


class TokenExclusion:
    """Exclude a native special token without fixing observed positions.

    A synthetic diffusion mask outside the model vocabulary has no AR class.
    """

    def __init__(self, token_id=None):
        self.token_id = token_id

    def exclude(self, logits):
        if self.token_id is None or self.token_id >= logits.shape[-1]:
            return logits
        return exclude_token(logits, self.token_id)

    def condition(self, log_probs, observed):
        return log_probs

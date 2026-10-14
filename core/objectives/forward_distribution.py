"""Distribution preparation and independent divergence terms."""

import torch
from torch import nn
from torch.distributions import kl_divergence
from torch.nn import functional as F


class TokenProbabilities(nn.Module):
    """Marginalize a discrete kernel over the predicted clean-token distribution."""

    def forward(self, log_probs, targets, representation):
        return F.one_hot(
            targets, log_probs.shape[-1]
        ).double(), log_probs.double().exp()


class CleanVector(nn.Module):
    """Use the predicted mean clean vector, not a mixture over clean tokens."""

    def forward(self, log_probs, targets, representation):
        return representation.encode(targets).double(), representation.posterior_mean(
            log_probs.exp()
        ).double()


class ForwardDistributions(nn.Module):
    outputs = {
        "target_distribution",
        "predicted_distribution",
        "observed",
        "coefficients",
        "process",
    }

    def __init__(self, prediction, view_index=0):
        super().__init__()
        if not isinstance(view_index, int) or view_index < 0:
            raise ValueError("view_index must be a nonnegative integer")
        self.prediction, self.view_index = prediction, view_index

    def forward(self, log_probs, targets, *, context):
        i = self.view_index
        if context is None or not (
            len(context.processes)
            == len(context.states)
            == len(context.coefficients)
            == len(context.representations)
            and i < len(context.processes)
        ):
            raise ValueError("forward distributions require a populated view context")
        process = context.processes[i]
        kernel = getattr(process, "forward_distribution", None)
        if not callable(kernel):
            raise TypeError("selected process must expose forward_distribution")
        clean, predicted = self.prediction(
            log_probs, targets, context.representations[i]
        )
        coefficients = context.coefficients[i]
        return dict(
            target_distribution=kernel(clean, coefficients),
            predicted_distribution=kernel(predicted, coefficients),
            observed=context.states[i],
            coefficients=coefficients,
            process=process,
        )


class ForwardKL(nn.Module):
    def forward(self, target, predicted):
        # Unsupported pairs deliberately raise NotImplementedError.
        return kl_divergence(target, predicted)


class ObservedRatioPenalty(nn.Module):
    def forward(self, target, predicted, observed):
        log_ratio = target.log_prob(observed) - predicted.log_prob(observed)
        return torch.expm1(log_ratio) - log_ratio

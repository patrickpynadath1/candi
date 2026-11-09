"""Conditional reveal/Gaussian ELBO with a forced-target proposal."""
import torch
from torch.nn import functional as F


class HybridELBO:
    def __call__(self, log_probs, tokens, clean, predicted, gamma, value, density,
                 lower, upper, masked, valid):
        ce = -log_probs.gather(-1, tokens[..., None]).squeeze(-1).double()
        mse = (predicted-clean).double().square().sum(-1)
        continuous = (0.5*value*(upper-lower)*torch.exp(-gamma))[:, None]*mse
        count = valid.sum(-1)
        masked_count = masked.sum(-1).clamp_min(1)
        estimate = ((ce+continuous)*masked).sum(-1)/masked_count/density.detach()
        interior = (ce+continuous)*masked*(count/masked_count/density.detach())[:, None]
        prior = 0.5*(torch.sigmoid(-upper)*clean.double().square().sum(-1)
                     + clean.shape[-1]*(torch.sigmoid(upper)-1-F.logsigmoid(upper)))
        return (interior+prior)*valid, estimate

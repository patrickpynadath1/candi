"""Pure tensor conversions for detached self-conditioning signals."""

import torch


def posterior_mean(probabilities, embedding_table):
    with torch.autocast(probabilities.device.type, enabled=False):
        return probabilities.float() @ embedding_table.float()


def final_hidden(hidden):
    # The architecture owns affine LayerNorm; retain the raw residual stream here.
    return hidden


SOURCES = {"posterior_mean": posterior_mean, "hidden": final_hidden}

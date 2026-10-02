"""Explicit tensor views for denoiser-only rows in likelihood training."""

import torch


def detach_selected(value, selected):
    mask = selected.reshape(len(selected), *((1,) * (value.ndim - 1)))
    return torch.where(mask, value.detach(), value)


def embedding_mean(probabilities, table, selected=None):
    with torch.autocast(probabilities.device.type, enabled=False):
        mean = probabilities.float() @ table.float()
        if selected is None:
            return mean
        detached_table_mean = probabilities.float() @ table.detach().float()
        return torch.where(selected[:, None, None], detached_table_mean, mean)

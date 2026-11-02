"""Append projected token states without reprojecting the existing prefix."""

import torch


def append_key_value(previous, current):
    if previous is None:
        return current
    return tuple(torch.cat([old, new], dim=1) for old, new in zip(previous, current))

"""Utilities for initializing spherical backbones from Duo checkpoints."""

import torch


def load_compatible_checkpoint(module, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = checkpoint.get("state_dict", checkpoint)
    target = module.state_dict()
    loaded = {}
    for name, value in source.items():
        if not name.startswith("backbone."):
            continue
        name = name.removeprefix("backbone.").replace(
            "vocab_embed.embedding", "sphere_embed.weight"
        )
        if name not in target:
            continue
        destination = target[name]
        if value.shape == destination.shape:
            loaded[name] = value
        elif value.ndim and value.shape[1:] == destination.shape[1:]:
            adapted = destination.clone()
            count = min(value.shape[0], destination.shape[0])
            adapted[:count] = value[:count]
            loaded[name] = adapted
    module.load_state_dict(loaded, strict=False)
    return tuple(sorted(loaded))

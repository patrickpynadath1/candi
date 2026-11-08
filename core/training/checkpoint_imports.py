"""Key renames for importing published checkpoints from other codebases.

Two external layouts are supported:

- a published MDLM checkpoint (``backbone.*`` keys of the original DiT), imported
  into an MDLM recipe whose token view holds the embedding table;
- an original CANDI checkpoint (a DiT backbone shared by several views).

Native checkpoints of this codebase load strictly and need no renaming.
"""

# Original DiT module names -> ModularDiT slot names.
_DIT_RENAMES = (
    ("sigma_map.", "time.embedder."),
    ("output_layer.", "output.layer."),
    ("conditioning_adapter.", "self_conditioning.adapter."),
)


def rename_dit_key(key, table="input.views.0.representation.embedding"):
    """One original DiT state key as the ModularDiT key (token table under ``table``)."""
    if key == "vocab_embed.embedding":
        return table
    for old, new in _DIT_RENAMES:
        if key.startswith(old):
            return new + key[len(old):]
    return key


def published_mdlm_state(state_dict):
    """``algorithm.model.*`` state of an MDLM recipe from a published checkpoint."""
    state = {}
    for key, value in state_dict.items():
        if not key.startswith("backbone."):
            continue
        rest = key.removeprefix("backbone.").replace("_orig_mod.", "")
        # Added to the original DiT after its no-AdaLN baseline; unused by MDLM.
        if rest == "position_rotary_emb.inv_freq":
            continue
        state["algorithm.model." + rename_dit_key(rest)] = value
    return state


def convert_hybrid_state_dict(state_dict):
    """Rename an original CANDI hybrid layout (``backbone.*``, ``views.*`` or the older
    ``branches.*``) to the view ModularDiT's keys."""
    converted = {}
    for key, value in state_dict.items():
        rest = key
        if rest.startswith("branches."):
            suffix = rest[len("branches."):]
            if ".adapter." in suffix:
                suffix = suffix.replace(".adapter.", ".representation.", 1)
            else:
                suffix = suffix.replace(".representation.", ".representation.state_representation.", 1)
            rest = "views." + suffix
        if rest.startswith("backbone.vocab_embed."):
            continue  # The same tensor as the token view's table.
        if rest.startswith("backbone.feedback_table_adapter."):
            rest = "input.feedback_table." + rest[len("backbone.feedback_table_adapter."):]
        elif rest.startswith("backbone."):
            rest = rename_dit_key(rest[len("backbone."):])
        elif rest.startswith("views."):
            rest = "input." + rest
        elif rest.startswith("conditioning_adapter."):
            rest = "self_conditioning.adapter." + rest[len("conditioning_adapter."):]
        converted[rest] = value
    return converted

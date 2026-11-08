"""Build docs/validation/compatibility.md: which samplers and modifiers each family accepts.

Every cell composes the family's small Sudoku preset with one sampler (or one
modifier), builds it, takes one training step and samples a few steps with a
tiny network on CPU. A cell is

- works: training and sampling ran;
- rejected: a contract check refused the combination (TypeError/ValueError);
  the message is the code's own explanation;
- error: anything else (a bug; the table makes it visible).

    uv run --locked python scripts/docs/compatibility_table.py [--output PATH]
"""

import argparse
import re
from pathlib import Path
import sys
import traceback

import torch
from omegaconf import OmegaConf, open_dict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FAMILIES = {
    "MDLM (masked)": "sudoku_hard",
    "Posterior uniform": "uniform_posterior",
    "Plug-in uniform": "uniform_diffusion",
    "DUO": "duo",
    "SEDD (absorbing)": "sedd_absorb",
    "D3PM (absorbing)": "d3pm_absorb",
    "GIDD": "gidd_elbo",
    "Gaussian": "gaussian_diffusion_sudoku",
    "RePlaid": "replaid_sudoku_hard",
    "Simplex": "sudoku_simplex",
    "Spherical": "sudoku_spherical",
    "Spherical vMF": "spherical_vmf",
    "Blurring": "sudoku_blurring_hard",
    "CANDI hybrid": "candi_hybrid",
    "Autoregressive": "ar",
    "Permuted AR": "permuted_autoregressive",
}
SAMPLERS = sorted(path.stem for path in (ROOT / "configs/inference/sampler").glob("*.yaml"))
MODIFIERS = {
    "self-conditioning": {"self_conditioning.enabled": True, "self_conditioning.probability": 1.0},
    "block diffusion": {"block_diffusion.enabled": True, "block_diffusion.block_size": 4},
    "token feedback": {"token_feedback.enabled": True, "token_feedback.depth": 0},
}
TINY = {
    "model.hidden_size": 16,
    "model.n_heads": 2,
    "model.n_blocks": 1,
    "model.dropout": 0.0,
}


def _set(cfg, key, value):
    if OmegaConf.select(cfg, key, default="__missing__") != "__missing__":
        OmegaConf.update(cfg, key, value, merge=False)


def build(experiment, sampler=None, overrides=None):
    from core.sweeps import compose

    cfg = compose(experiment, {})
    for key, value in {**TINY, **(overrides or {})}.items():
        _set(cfg, key, value)
    _set(cfg, "model.cond_dim", 8)
    views = OmegaConf.select(cfg, "model.views", default=None)
    names = range(len(views)) if OmegaConf.is_list(views) else (views or ())
    for name in names:
        _set(cfg, f"model.views.{name}.representation.hidden_size", TINY["model.hidden_size"])
    if sampler is not None:
        cfg.sampler = compose(overrides={"inference/sampler": sampler}).sampler
    length = int(OmegaConf.select(cfg, "task.length"))
    # Entropy-bounded samplers must be able to finish the whole sequence.
    bounded = "EntropyBounded" in str(OmegaConf.select(cfg, "sampler._target_", default=""))
    _set(cfg, "sampler.steps", length if bounded else 3)
    _set(cfg, "sampler.max_steps", length)
    with open_dict(cfg):
        cfg.training.ema = 0
    return cfg


def run_cell(experiment, sampler=None, overrides=None):
    from hydra.utils import instantiate

    from core.training.module import TrainingModule

    torch.manual_seed(0)
    cfg = build(experiment, sampler, overrides)
    module = TrainingModule(cfg)
    algorithm = module.algorithm
    length = int(OmegaConf.select(cfg, "task.length"))
    vocab = int(OmegaConf.select(cfg, "task.vocab_size"))
    tokens = torch.randint(0, vocab, (2, length))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    mask[:, : max(1, length // 4)] = False
    algorithm.train()
    algorithm.loss(tokens, mask, training=True)["loss"].backward()
    algorithm.eval()
    # A one-token prompt (autoregressive models need a start), or none for
    # recipes trained unconditionally, which refuse prompts.
    prompt = torch.zeros_like(mask)
    prompt[:, 0] = True
    try:
        algorithm.generate(instantiate(cfg.sampler), tokens, prompt)
    except (TypeError, ValueError) as error:
        if "prompt" not in str(error):
            raise
        algorithm.generate(instantiate(cfg.sampler), tokens, torch.zeros_like(mask))


def outcome(experiment, sampler=None, overrides=None):
    try:
        run_cell(experiment, sampler, overrides)
        return "works", ""
    except Exception as error:  # noqa: BLE001 - report every failure
        cause = error
        while not isinstance(cause, (TypeError, ValueError)) and cause.__cause__ is not None:
            cause = cause.__cause__  # Hydra wraps constructor errors
        if isinstance(cause, (TypeError, ValueError)):
            message = str(cause).splitlines()[0][:160] if str(cause) else type(cause).__name__
            # One note per reason, not per sampler class that states it.
            message = re.sub(r"^[A-Z]\w*Sampler requires", "this sampler requires", message)
            return "rejected", message
        detail = traceback.format_exception_only(type(error), error)[-1].strip()
        return "error", detail[:160]


def table(rows, columns, cells, notes):
    symbols = {"works": "✓", "rejected": "–", "error": "✗"}
    lines = ["| | " + " | ".join(f"`{c}`" for c in columns) + " |",
             "| --- | " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        entries = []
        for column in columns:
            status, message = cells[row, column]
            mark = symbols[status]
            if message:
                note = notes.setdefault(message, len(notes) + 1)
                mark += f"<sup>{note}</sup>"
            entries.append(mark)
        lines.append(f"| {row} | " + " | ".join(entries) + " |")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=ROOT / "docs/validation/compatibility.md")
    args = parser.parse_args()
    notes = {}
    sampler_cells = {}
    for family, experiment in FAMILIES.items():
        for sampler in SAMPLERS:
            sampler_cells[family, sampler] = outcome(experiment, sampler)
            print(family, sampler, sampler_cells[family, sampler][0], flush=True)
    modifier_cells = {}
    for family, experiment in FAMILIES.items():
        for name, overrides in MODIFIERS.items():
            modifier_cells[family, name] = outcome(experiment, None, overrides)
            print(family, name, modifier_cells[family, name][0], flush=True)
    body = [
        "# Compatibility of families, samplers and modifiers",
        "",
        "Generated by `scripts/docs/compatibility_table.py`; do not edit by hand. Each cell",
        "builds the family's small Sudoku preset with one sampler (or modifier), takes one",
        "training step and samples three steps on CPU from a one-token prompt.",
        "",
        "✓ works · – rejected by a contract check (reason in the notes) · ✗ error",
        "",
        "## Samplers",
        "",
        *table(list(FAMILIES), SAMPLERS, sampler_cells, notes),
        "",
        "## Modifiers (with the family's own sampler)",
        "",
        *table(list(FAMILIES), list(MODIFIERS), modifier_cells, notes),
        "",
        "## Notes",
        "",
        *[f"{number}. {message}" for message, number in notes.items()],
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(body))
    errors = [k for k, v in {**sampler_cells, **modifier_cells}.items() if v[0] == "error"]
    print(f"wrote {args.output}; {len(errors)} error cells")


if __name__ == "__main__":
    main()

# AGENTS.md

Instructions for coding agents working in this repository. People should start at
[README.md](README.md); contributors at [CONTRIBUTING.md](CONTRIBUTING.md).

SDDC (Some DLM Design Choices) is a framework for controlled experiments on diffusion
language models. A model is a **recipe** with three parts, and the code and configs follow
them ([docs/concepts/recipe.md](docs/concepts/recipe.md)):

- **information path**: how tokens are represented and corrupted (representation, process,
  noise schedule);
- **denoising task**: what the network predicts and how it is trained (model,
  parameterization, objective, training time sampler);
- **inference settings**: how a trained model generates (sampler, steps, commitment,
  terminal decoding).

Modifiers (self-conditioning, block diffusion, hybrid views, token feedback, variable length)
cut across the parts. Library code is in `core/`, Hydra configs in `configs/` (layout in
[configs/README.md](configs/README.md)), docs in `docs/` (start at [docs/index.md](docs/index.md)),
and the blog post's worked examples in `experiments/`.

## Skills

Each task below has a playbook in `.claude/skills/<name>/SKILL.md`. Read the matching one
before starting; each ends with the check that shows the work is done.

| Skill | Use it to |
| --- | --- |
| [sddc-setup](.claude/skills/sddc-setup/SKILL.md) | install the environment, prepare data, and confirm everything works |
| [sddc-new-recipe](.claude/skills/sddc-new-recipe/SKILL.md) | compose or change a recipe from existing parts and check it |
| [sddc-sweep](.claude/skills/sddc-sweep/SKILL.md) | write a sweep config and prepare a plan (manifest) of cases |
| [sddc-run](.claude/skills/sddc-run/SKILL.md) | run, resume and monitor nested tuning |
| [sddc-remote](.claude/skills/sddc-remote/SKILL.md) | drive a GPU server from a laptop through Ray Jobs |
| [sddc-analyze](.claude/skills/sddc-analyze/SKILL.md) | turn run trees into tables and compare them with expected numbers |
| [sddc-archive](.claude/skills/sddc-archive/SKILL.md) | pack finished runs into a portable archive and verify it |
| [sddc-add-task](.claude/skills/sddc-add-task/SKILL.md) | add a dataset or task with its metrics and evaluation splits |
| [sddc-add-information-path](.claude/skills/sddc-add-information-path/SKILL.md) | add a representation, corruption process, noise schedule or hybrid view |
| [sddc-add-denoising-component](.claude/skills/sddc-add-denoising-component/SKILL.md) | add a parameterization, objective, time sampler or algorithm |
| [sddc-add-architecture](.claude/skills/sddc-add-architecture/SKILL.md) | change the network: model slots, block sublayers, depth mixing, encoder/decoder blocks |
| [sddc-add-inference](.claude/skills/sddc-add-inference/SKILL.md) | add a sampler, update rule, commitment policy or sweep inference policy |
| [sddc-add-modifier](.claude/skills/sddc-add-modifier/SKILL.md) | add or change a cross-cutting modifier |

## Commands

```bash
uv sync --locked --extra tuning --extra scoring --extra tracking --extra mazes
uv run --locked python scripts/doctor.py --smoke          # environment, data and a 2-step CPU run
uv run --locked pytest -q core/tests                      # full suite
uv run --locked ruff check core main.py frontier.py scripts experiments
uv run --locked python main.py +experiment=<preset> --cfg job   # print a composed config
```

The lock pins PyTorch for CUDA 12.1 on Linux x86_64; on other platforms see the
`sddc-setup` skill. Use `uv run --locked`, never system Python. The default `main.py` (no
overrides) is a full GPU training run; don't launch it casually.

## Rules

- **Shared machines.** Never stop clusters, run a bare `ray stop`, kill jobs, or delete
  caches, checkpoints or results you did not create, without the user's explicit go-ahead.
  Check live state instead of trusting notes. Launching GPU jobs needs the user's go-ahead.
- **Storage.** Everything persistent (data, plans, runs, archives) goes under
  `SDDC_STORAGE_ROOT` (default `~/.cache/sddc`). Never commit datasets, checkpoints, run
  outputs, credentials or `.venv`.
- **Protocols.** Selection uses validation data only; the test split is scored once, for the
  winner. Changing an existing experiment's protocol needs new case and output names;
  `nested_tuning` refuses to resume a case whose protocol changed.
- **Recipes.** Changing only `denoising_task/algorithm=` does not swap the matching process,
  model, schedule or sampler. Switch families with a preset from `configs/experiment/` and
  check it with `--cfg job`.
- **Boundaries.** Algorithms assemble losses and do not own the reverse-time loop; samplers
  own time progression, stopping and NFE accounting (NFE counts every network call, including
  terminal ones); the training module owns optimization, not diffusion equations. Read
  [docs/concepts/architecture.md](docs/concepts/architecture.md) and
  [docs/concepts/component-contracts.md](docs/concepts/component-contracts.md) before moving a
  boundary.
- **Masks.** `loss_mask` = contributes to the loss, `corruption_mask` = may be corrupted,
  `prompt_mask` True = fixed during generation.
- **Changes.** A new component is a YAML option in its `configs/` group, a doc page, and tests
  of an independent property (not a restatement of the formula). Prefer a new recipe to a new
  algorithm class. Refactors keep behavior, not bytes: tolerance-based comparisons are fine.
  Regenerate `docs/validation/compatibility.md` when you change what a family accepts.

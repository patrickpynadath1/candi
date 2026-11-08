---
name: sddc-new-recipe
description: Compose a new SDDC recipe or change one part of an existing recipe (sampler, schedule, objective, view, modifier) from existing components, and check that it composes, trains and samples. Use before writing new code, when asked to "try X with Y", or when adding a preset to configs/experiment/.
---

# Compose or change a recipe

Most new methods are recipes, not code. Start from the closest preset in
`configs/experiment/` (the paper presets are listed in `docs/guides/paper-recipes.md`) and
change one part at a time.

## 1. Find the parts

- Layout: `configs/README.md`. Groups live under `information_path/`, `denoising_task/`,
  `inference/`, `modifiers/` and `training/`; their config keys stay short (`sampler`,
  `process`, `model`, ...).
- What a family accepts: `docs/validation/compatibility.md` (generated; every family x
  sampler x modifier cell is built and run).
- What each component means: `docs/components/` and `docs/methods/`.

## 2. Compose

```bash
# One part swapped on the command line:
uv run --locked python main.py +experiment=mdlm inference/sampler=ancestral sampler.steps=128 --cfg job
# A hybrid view swapped (views carry an explicit package):
uv run --locked python main.py +experiment=corruption_factorial \
  information_path/hybrid_view@model.views.content=simplex --cfg job
```

A group override selects a file (`inference/sampler=ancestral` →
`configs/inference/sampler/ancestral.yaml`); a dotted override sets a field
(`sampler.steps=128`). Inside a preset, use absolute paths:
`- override /information_path/process: gaussian`. A model slot keeps its package:
`- override /denoising_task/model/input@model.input: linear`.

Changing only `denoising_task/algorithm=` does **not** swap the matching process, model,
schedule or sampler. To switch families, copy a verified preset of that family.

## 3. Save it as a preset (when it will be reused)

`configs/experiment/<name>.yaml` starts with `# @package _global_`, lists `override` entries
for each group it changes, then `_self_`, then field values. Copy the nearest preset rather
than starting empty. Add one line to the relevant `docs/methods/` page.

## 4. Check

1. `--cfg job`: every `_target_` is the class you meant, and nothing from the old family is
   left (process, schedule, sampler, parameterization agree).
2. A tiny CPU run trains, validates and samples (the sizes of `configs/experiment/smoke.yaml`;
   two presets cannot be stacked, so pass them as overrides):

   ```bash
   uv run --locked python main.py +experiment=<name> \
     model.hidden_size=32 model.cond_dim=16 model.n_blocks=1 model.n_heads=4 \
     trainer.accelerator=cpu trainer.devices=1 trainer.precision=32-true \
     trainer.max_steps=2 trainer.val_check_interval=2 trainer.num_sanity_val_steps=0 \
     data.num_train=8 data.num_valid=2 loader.global_batch_size=4 loader.eval_batch_size=2 \
     loader.num_workers=0 loader.pin_memory=false checkpoint.every_n_steps=1 \
     callback.every_n_steps=0 evaluation.device=cpu evaluation.max_examples=2 \
     hydra.run.dir=$SDDC_STORAGE_ROOT/scratch/<name>
   ```

   or, in a test, `core.sweeps.configuration.compose("<name>", [overrides])` with a tiny model
   and one `algorithm.loss(...)` plus one sampler call (pattern:
   `core/tests/test_continuous_adaptive_feedback.py`).
3. If the recipe is a new family or changes what a family accepts, regenerate the table:
   `uv run --locked python scripts/docs/compatibility_table.py`, and commit the diff.

## Done when

`--cfg job` shows the intended components, the tiny run completes, and
`core/tests/test_run_preparation.py` (every preset composes) passes.

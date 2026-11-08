# Configs

Hydra configs, laid out like a recipe ([docs/concepts/recipe.md](../docs/concepts/recipe.md)).
`config.yaml` is the root for `main.py`; `frontier.yaml` is the root for
`python -m core.sweeps.frontier`.

| Folder | What it holds | Groups |
| --- | --- | --- |
| `information_path/` | how tokens are represented and corrupted | `representation`, `process`, `noise_schedule`, `token_space`, `time_map`; the hybrid views `hybrid_view` and the sweep pieces built from them (`content_view`, `discrete_view`, `gaussian_state`, `path_schedule`) |
| `denoising_task/` | what the network predicts and how it is trained | `algorithm`, `model` (with `input/` and `output/` slots), `parameterization`, `prediction`, `objective`, `time_sampler`; model variants `block`, `block_sublayer`, `depth_mixing`; DUO's `curriculum` and `calibration` |
| `inference/` | how a trained model generates | `sampler`, `inference_policies` (the per-sampler tuning spaces of a sweep), `generation_order`, `evaluator`, `frontier` |
| `modifiers/` | switches that cut across the three parts | `sc` and `feedback` (self-conditioning), `variable_length` |
| `training/` | the training loop | `datamodule`, `callback`, `lr_scheduler`, `strategy` |
| `task/`, `data/` | tasks, task suites (`task/battery/`) and dataset settings | |
| `experiment/` | full recipe presets: `+experiment=mdlm` | |
| `sweep/` | one sweep per worked example plus general method sweeps; `common.yaml` holds their shared protocol | |
| `ablation_path/`, `ladder_recipe/` | the arms the worked-example sweeps compare | |

A group's path is how you select it; its config key stays short:

```bash
uv run --locked python main.py +experiment=mdlm inference/sampler=ancestral sampler.steps=128
uv run --locked python main.py +experiment=duo denoising_task/objective=weighted_ce
```

Inside a preset, override with the absolute path: `- override /information_path/process: gaussian`.
A model slot keeps its package in the override: `- override /denoising_task/model/input@model.input: linear`.

Changing only `denoising_task/algorithm=` does not swap the matching process, model, schedule or
sampler. Switch families with a preset from `experiment/`, and check it with `--cfg job`.

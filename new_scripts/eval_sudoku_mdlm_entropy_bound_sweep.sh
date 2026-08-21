#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 GPU_ID [NUM_EXAMPLES]" >&2
  exit 2
fi

gpu_id=$1
num_examples=${2:-128}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checkpoint="${repo_root}/new_sudoku_runs/mdlm/checkpoints/last.ckpt"
result_root="${repo_root}/new_eval_results/sudoku_entropy_bound"

if [[ ! -f "${checkpoint}" ]]; then
  echo "missing checkpoint: ${checkpoint}" >&2
  exit 1
fi

mkdir -p "${result_root}"

baseline_dir="${result_root}/ancestral_n${num_examples}"
if [[ ! -f "${baseline_dir}/results.json" ]]; then
  echo "evaluating ancestral baseline on ${num_examples} puzzles"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
    --config-name=config \
    +experiment=sudoku_mdlm \
    mode=sudoku_eval \
    sampling.predictor=ancestral_cache \
    sampling.steps=180 \
    "sudoku.max_examples=${num_examples}" \
    sudoku.batch_size=64 \
    loader.eval_global_batch_size=64 \
    loader.num_workers=1 \
    "eval.checkpoint_path=${checkpoint}" \
    "sudoku.output_dir=${baseline_dir}" \
    "hydra.run.dir=${baseline_dir}" \
    wandb=null \
    >"${result_root}/ancestral_n${num_examples}.log" 2>&1
fi

for gamma in 0 0.01 0.05 0.1 0.25 0.5 1 2 5; do
  label=${gamma//./p}
  output_dir="${result_root}/gamma_${label}_n${num_examples}"
  if [[ -f "${output_dir}/results.json" ]]; then
    echo "skipping completed entropy bound gamma=${gamma}"
    continue
  fi
  echo "evaluating entropy bound gamma=${gamma} on ${num_examples} puzzles"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${repo_root}/.venv/bin/python" "${repo_root}/main.py" \
    --config-name=config \
    +experiment=sudoku_mdlm \
    mode=sudoku_eval \
    sampling.predictor=entropy_bounded \
    "sampling.entropy_bound=${gamma}" \
    sampling.steps=180 \
    "sudoku.max_examples=${num_examples}" \
    sudoku.batch_size=64 \
    loader.eval_global_batch_size=64 \
    loader.num_workers=1 \
    "eval.checkpoint_path=${checkpoint}" \
    "sudoku.output_dir=${output_dir}" \
    "hydra.run.dir=${output_dir}" \
    wandb=null \
    >"${result_root}/gamma_${label}_n${num_examples}.log" 2>&1
done

"${repo_root}/.venv/bin/python" - "${result_root}" "${num_examples}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
count = sys.argv[2]
summary = []
baseline_path = root / f"ancestral_n{count}" / "results.json"
if baseline_path.exists():
    result = json.loads(baseline_path.read_text())
    summary.append({
        "sampler": "ancestral_cache",
        "gamma": None,
        "accuracy": result["accuracy"],
        "cell_accuracy": result["cell_accuracy"],
        "valid_grid_rate": result["valid_grid_rate"],
        "mean_nfe": result["mean_nfe"],
        "num_total": result["num_total"],
    })
for path in sorted(root.glob(f"gamma_*_n{count}/results.json")):
    result = json.loads(path.read_text())
    summary.append({
        "sampler": "entropy_bounded",
        "gamma": path.parent.name.split("_n", 1)[0].removeprefix("gamma_").replace("p", "."),
        "accuracy": result["accuracy"],
        "cell_accuracy": result["cell_accuracy"],
        "valid_grid_rate": result["valid_grid_rate"],
        "mean_nfe": result["mean_nfe"],
        "num_total": result["num_total"],
    })
destination = root / f"summary_n{count}.json"
destination.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

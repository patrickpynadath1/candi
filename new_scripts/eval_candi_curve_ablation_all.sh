#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
launcher="${repo_root}/new_scripts/eval_candi_curve_ablation_one.sh"
gpus=(5 6 7)
specs=(
  "legacy_fixed|sudoku_candi_curve_5k|algo.curve_type=linear algo.curve_orientation=linear"
  "linear_adaptive|sudoku_candi_curve_adaptive_5k|algo.curve_type=linear algo.curve_orientation=linear"
  "factorized_joint|sudoku_candi_factorized_probe_5k|"
  "continuous_first_p2|sudoku_candi_curve_5k|algo.curve_type=power algo.curve_orientation=continuous_first algo.curve_power=2"
  "continuous_first_p4|sudoku_candi_curve_5k|algo.curve_type=power algo.curve_orientation=continuous_first algo.curve_power=4"
  "continuous_first_p8|sudoku_candi_curve_5k|algo.curve_type=power algo.curve_orientation=continuous_first algo.curve_power=8"
  "discrete_first_p2|sudoku_candi_curve_5k|algo.curve_type=power algo.curve_orientation=discrete_first algo.curve_power=2"
  "discrete_first_p4|sudoku_candi_curve_5k|algo.curve_type=power algo.curve_orientation=discrete_first algo.curve_power=4"
  "bezier_continuous|sudoku_candi_curve_5k|algo.curve_type=bezier algo.curve_control_1=[0.05,0.55] algo.curve_control_2=[0.45,0.95]"
  "bezier_discrete|sudoku_candi_curve_5k|algo.curve_type=bezier algo.curve_control_1=[0.55,0.05] algo.curve_control_2=[0.95,0.45]"
  "sequential_continuous|sudoku_candi_curve_5k|algo.curve_type=sequential algo.curve_discrete_center=0.7 algo.curve_continuous_center=0.3 algo.curve_sharpness=10"
  "sequential_discrete|sudoku_candi_curve_5k|algo.curve_type=sequential algo.curve_discrete_center=0.3 algo.curve_continuous_center=0.7 algo.curve_sharpness=10"
  "continuous_first_p4_adaptive|sudoku_candi_curve_adaptive_5k|algo.curve_type=power algo.curve_orientation=continuous_first algo.curve_power=4"
  "discrete_first_p4_adaptive|sudoku_candi_curve_adaptive_5k|algo.curve_type=power algo.curve_orientation=discrete_first algo.curve_power=4"
  "bezier_continuous_adaptive|sudoku_candi_curve_adaptive_5k|algo.curve_type=bezier algo.curve_control_1=[0.05,0.55] algo.curve_control_2=[0.45,0.95]"
)
mkdir -p "${repo_root}/new_candi_curve_results"
pids=()
names=()
status=0
for index in "${!specs[@]}"; do
  if (( index >= 9 )); then
    slot=$((index - 9))
    if ! wait "${pids[slot]}"; then status=1; fi
  fi
  IFS='|' read -r name experiment overrides <<<"${specs[index]}"
  read -r -a override_args <<<"${overrides}"
  "${launcher}" "${gpus[index%3]}" "${name}" "${experiment}" \
    "${override_args[@]}" >"${repo_root}/new_candi_curve_results/${name}.eval.log" 2>&1 &
  pids+=("$!")
  names+=("${name}")
done
completed_slots=$((${#specs[@]} - 9))
for index in "${!pids[@]}"; do
  if (( index < completed_slots )); then continue; fi
  if wait "${pids[index]}"; then
    echo "CANDI_CURVE_EVAL_OK:${names[index]}"
  else
    echo "CANDI_CURVE_EVAL_FAILED:${names[index]}" >&2
    status=1
  fi
done
exit "${status}"

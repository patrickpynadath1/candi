#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
launcher="${repo_root}/new_scripts/eval_candi_temperature_one.sh"
if [[ -n "${CANDI_TEMPERATURES:-}" ]]; then
  read -r -a temperatures <<<"${CANDI_TEMPERATURES}"
else
  temperatures=(0.40 0.55 0.70 0.85 1.15)
fi
gpus=(5 6 7)
specs=(
  "legacy_fixed|sudoku_candi_curve_5k|algo.curve_type=linear algo.curve_orientation=linear"
  "bezier_discrete|sudoku_candi_curve_5k|algo.curve_type=bezier algo.curve_control_1=[0.55,0.05] algo.curve_control_2=[0.95,0.45]"
  "linear_adaptive|sudoku_candi_curve_adaptive_5k|algo.curve_type=linear algo.curve_orientation=linear"
  "continuous_first_p2|sudoku_candi_curve_5k|algo.curve_type=power algo.curve_orientation=continuous_first algo.curve_power=2"
  "bezier_continuous|sudoku_candi_curve_5k|algo.curve_type=bezier algo.curve_control_1=[0.05,0.55] algo.curve_control_2=[0.45,0.95]"
  "factorized_joint_raw_ce|sudoku_candi_factorized_probe_5k|"
)

mkdir -p "${repo_root}/new_candi_temperature_results/logs"
pids=()
names=()
status=0
index=0
for spec in "${specs[@]}"; do
  IFS='|' read -r name experiment overrides <<<"${spec}"
  read -r -a override_args <<<"${overrides}"
  for temperature in "${temperatures[@]}"; do
    # Keep at most three evaluations resident on each GPU.
    if (( index >= 9 )); then
      slot=$((index % 9))
      if ! wait "${pids[slot]}"; then status=1; fi
    else
      slot=$index
    fi
    gpu=${gpus[index%3]}
    "${launcher}" "${gpu}" "${name}" "${experiment}" "${temperature}" \
      "${override_args[@]}" \
      >"${repo_root}/new_candi_temperature_results/logs/${name}_temp_${temperature}.log" 2>&1 &
    pids[slot]=$!
    names[slot]="${name}@${temperature}"
    index=$((index + 1))
  done
done

for slot in "${!pids[@]}"; do
  if wait "${pids[slot]}"; then
    echo "CANDI_TEMP_EVAL_OK:${names[slot]}"
  else
    echo "CANDI_TEMP_EVAL_FAILED:${names[slot]}" >&2
    status=1
  fi
done
exit "${status}"

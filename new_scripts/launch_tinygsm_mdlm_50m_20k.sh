#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/patrick/duo"
exec "${repo_root}/new_scripts/launch_tinygsm_mdlm_50m_150k.sh" "$@"

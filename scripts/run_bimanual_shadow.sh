#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 VIDEO HDF5 OUTPUT [--force] [--start-stage STAGE] [--stop-stage STAGE] [--left-arm-reference-qpos Q1 ... Q6] [--right-arm-reference-qpos Q1 ... Q6]" >&2
  exit 2
fi

VIDEO="$1"
HDF5="$2"
RUN_ROOT="$3"
shift 3

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_ROOT="${EGODEX_DEXHAND_ROOT:-$(dirname "${PROJECT_ROOT}")}"
VENV_ROOT="${EGODEX_VENV_ROOT:-${PIPELINE_ROOT}/.venv}"
THIRD_PARTY_ROOT="${EGODEX_THIRD_PARTY_ROOT:-${PIPELINE_ROOT}/third_party}"
STOP_STAGE="compose"

EXTRA_ARGS=("$@")
index=0
while [[ ${index} -lt ${#EXTRA_ARGS[@]} ]]; do
  argument="${EXTRA_ARGS[${index}]}"
  case "${argument}" in
    --force)
      ;;
    --start-stage)
      index=$((index + 1))
      [[ ${index} -lt ${#EXTRA_ARGS[@]} ]] || { echo "missing --start-stage value" >&2; exit 2; }
      ;;
    --start-stage=*)
      ;;
    --stop-stage)
      index=$((index + 1))
      [[ ${index} -lt ${#EXTRA_ARGS[@]} ]] || { echo "missing --stop-stage value" >&2; exit 2; }
      STOP_STAGE="${EXTRA_ARGS[${index}]}"
      ;;
    --stop-stage=*)
      STOP_STAGE="${argument#*=}"
      ;;
    --left-arm-reference-qpos|--right-arm-reference-qpos)
      for _ in 1 2 3 4 5 6; do
        index=$((index + 1))
        [[ ${index} -lt ${#EXTRA_ARGS[@]} ]] || { echo "missing ${argument} values" >&2; exit 2; }
      done
      ;;
    --left-hide-arm-visual-link|--right-hide-arm-visual-link|--render-device)
      index=$((index + 1))
      [[ ${index} -lt ${#EXTRA_ARGS[@]} ]] || { echo "missing ${argument} value" >&2; exit 2; }
      ;;
    *)
      echo "unsupported runner argument: ${argument}" >&2
      exit 2
      ;;
  esac
  index=$((index + 1))
done

source "${VENV_ROOT}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/src:${THIRD_PARTY_ROOT}/sam2:${PYTHONPATH:-}"

python -m egodex_dexhand.bimanual_cli \
  --video "${VIDEO}" \
  --hdf5 "${HDF5}" \
  --output "${RUN_ROOT}" \
  --dex-assets "${THIRD_PARTY_ROOT}/dex-retargeting/assets/robots/hands" \
  --left-combined-urdf "${THIRD_PARTY_ROOT}/dex-retargeting/assets/robots/assembly/ur5e_shadow/ur5e_shadow_left_hand_glb.urdf" \
  --right-combined-urdf "${THIRD_PARTY_ROOT}/dex-retargeting/assets/robots/assembly/ur5e_shadow/ur5e_shadow_right_hand_glb.urdf" \
  --sam2-root "${THIRD_PARTY_ROOT}/sam2" \
  --sam2-checkpoint "${THIRD_PARTY_ROOT}/sam2/checkpoints/sam2.1_hiera_small.pt" \
  --sam2-size small \
  --propainter-root "${THIRD_PARTY_ROOT}/ProPainter" \
  --scale 0.5 \
  --prompt-stride "${EGODEX_PROMPT_STRIDE:-1}" \
  --smoothing-window 9 \
  --smoothing-passes 2 \
  --forearm-max-angular-velocity 2.0 \
  "$@"

if [[ "${STOP_STAGE}" == "compose" ]]; then
  python -m egodex_dexhand.verify "${RUN_ROOT}"
else
  echo "partial run stopped after ${STOP_STAGE}; full-output verification skipped"
fi

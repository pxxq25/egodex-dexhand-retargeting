#!/usr/bin/env bash
# Source this file after bootstrap: source ./activate.sh

_egodex_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EGODEX_DEXHAND_ROOT="${EGODEX_DEXHAND_ROOT:-$(dirname "${_egodex_project_root}")}"
export EGODEX_VENV_ROOT="${EGODEX_VENV_ROOT:-${EGODEX_DEXHAND_ROOT}/.venv}"
export EGODEX_THIRD_PARTY_ROOT="${EGODEX_THIRD_PARTY_ROOT:-${EGODEX_DEXHAND_ROOT}/third_party}"
export EGODEX_MEDIAPIPE_PYTHON="${EGODEX_MEDIAPIPE_PYTHON:-${EGODEX_DEXHAND_ROOT}/.venv-mediapipe/bin/python}"

# shellcheck source=/dev/null
source "${EGODEX_VENV_ROOT}/bin/activate"
export PYTHONPATH="${_egodex_project_root}/src:${_egodex_project_root}/scripts:${EGODEX_THIRD_PARTY_ROOT}/sam2:${EGODEX_THIRD_PARTY_ROOT}/sam3:${EGODEX_THIRD_PARTY_ROOT}/ProPainter:${PYTHONPATH:-}"
unset _egodex_project_root

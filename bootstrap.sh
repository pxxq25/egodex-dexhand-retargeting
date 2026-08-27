#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="${EGODEX_DEXHAND_ROOT:-$(dirname "${PROJECT_ROOT}")}"
PYTHON_COMMAND="${PYTHON_COMMAND:-python3}"
INSTALL_SYSTEM_PACKAGES=1
OFFLINE=0
WITHOUT_MODELS=0
DRY_RUN=0
RENDER_DEVICE=""

usage() {
  cat <<'USAGE'
Usage: ./bootstrap.sh [options]

Creates the complete pinned runtime, clones exact third-party revisions,
downloads and verifies checkpoints, and validates CUDA/Vulkan when available.

Options:
  --runtime PATH          runtime root (default: repository parent)
  --python COMMAND        CPython 3.10 executable (default: python3)
  --offline               use existing wheel/git/model caches only
  --without-models        install code only; do not download checkpoints
  --skip-system-packages  require, but do not apt-install, system dependencies
  --render-device DEVICE  require a SAPIEN preflight, e.g. cuda:0
  --dry-run               print third-party/model actions without changing state
  -h, --help              show this help

For the gated SAM3 checkpoint, request access to facebook/sam3 on Hugging Face
and run: HF_TOKEN=hf_... ./bootstrap.sh
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime)
      [[ $# -ge 2 ]] || { echo "missing --runtime value" >&2; exit 2; }
      RUNTIME_ROOT="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || { echo "missing --python value" >&2; exit 2; }
      PYTHON_COMMAND="$2"
      shift 2
      ;;
    --offline)
      OFFLINE=1
      shift
      ;;
    --without-models)
      WITHOUT_MODELS=1
      shift
      ;;
    --skip-system-packages)
      INSTALL_SYSTEM_PACKAGES=0
      shift
      ;;
    --render-device)
      [[ $# -ge 2 ]] || { echo "missing --render-device value" >&2; exit 2; }
      RENDER_DEVICE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

log() {
  printf '\n==> %s\n' "$*"
}

die() {
  echo "bootstrap error: $*" >&2
  exit 1
}

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  "$@"
}

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  die "the production runtime is locked for Linux x86_64"
fi

command -v "${PYTHON_COMMAND}" >/dev/null 2>&1 || die "${PYTHON_COMMAND} not found"
"${PYTHON_COMMAND}" - <<'PY' || die "CPython 3.10 is required"
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)
PY

RUNTIME_ROOT="$("${PYTHON_COMMAND}" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${RUNTIME_ROOT}")"

if [[ ${DRY_RUN} -eq 1 ]]; then
  run "${PYTHON_COMMAND}" "${PROJECT_ROOT}/scripts/setup_runtime.py" \
    --runtime "${RUNTIME_ROOT}" \
    --project "${PROJECT_ROOT}" \
    --python "${RUNTIME_ROOT}/.venv/bin/python" \
    --dry-run \
    $([[ ${OFFLINE} -eq 1 ]] && printf '%s' --offline) \
    $([[ ${WITHOUT_MODELS} -eq 1 ]] && printf '%s' --without-models)
  exit 0
fi

missing_commands=()
for command in git curl ffmpeg vulkaninfo; do
  command -v "${command}" >/dev/null 2>&1 || missing_commands+=("${command}")
done

if [[ ${#missing_commands[@]} -gt 0 && ${INSTALL_SYSTEM_PACKAGES} -eq 1 ]]; then
  command -v apt-get >/dev/null 2>&1 || die \
    "missing ${missing_commands[*]} and apt-get is unavailable"
  if [[ ${EUID} -eq 0 ]]; then
    apt_prefix=()
  elif command -v sudo >/dev/null 2>&1; then
    apt_prefix=(sudo)
  else
    die "missing ${missing_commands[*]}; rerun as root or install them manually"
  fi
  log "Installing Ubuntu runtime packages"
  run "${apt_prefix[@]}" apt-get update
  run "${apt_prefix[@]}" apt-get install -y --no-install-recommends \
    ca-certificates curl ffmpeg git libgl1 libglib2.0-0 libvulkan1 \
    python3-venv vulkan-tools
fi

for command in git curl ffmpeg vulkaninfo; do
  command -v "${command}" >/dev/null 2>&1 || die "required command missing: ${command}"
done

mkdir -p "${RUNTIME_ROOT}"
available_kib="$(df -Pk "${RUNTIME_ROOT}" | awk 'NR==2 {print $4}')"
if [[ -n "${available_kib}" && ${available_kib} -lt 12582912 ]]; then
  die "at least 12 GiB of free space is required under ${RUNTIME_ROOT}"
fi

VENV="${RUNTIME_ROOT}/.venv"
MEDIAPIPE_VENV="${RUNTIME_ROOT}/.venv-mediapipe"
pip_network_args=(--retries 8 --timeout 120)
if [[ ${OFFLINE} -eq 1 ]]; then
  pip_network_args+=(--no-index)
fi

log "Creating the pinned Python 3.10 environments"
if [[ ! -x "${VENV}/bin/python" ]]; then
  run "${PYTHON_COMMAND}" -m venv "${VENV}"
fi
if [[ ! -x "${MEDIAPIPE_VENV}/bin/python" ]]; then
  run "${PYTHON_COMMAND}" -m venv "${MEDIAPIPE_VENV}"
fi

for environment_python in "${VENV}/bin/python" "${MEDIAPIPE_VENV}/bin/python"; do
  run "${environment_python}" -m pip install "${pip_network_args[@]}" \
    pip==25.2 setuptools==80.9.0 wheel==0.45.1
done

log "Installing the fully pinned CUDA 12.6 production environment"
run "${VENV}/bin/python" -m pip install "${pip_network_args[@]}" --no-deps \
  --no-build-isolation \
  -r "${PROJECT_ROOT}/requirements/runtime-cu126.lock.txt"
run "${VENV}/bin/python" -m pip install "${pip_network_args[@]}" --no-deps \
  -r "${PROJECT_ROOT}/requirements/pytorch-cu126.lock.txt"
run "${VENV}/bin/python" -m pip install --no-deps --no-build-isolation \
  -e "${PROJECT_ROOT}"

log "Installing the isolated MediaPipe RGB detector"
run "${MEDIAPIPE_VENV}/bin/python" -m pip install "${pip_network_args[@]}" \
  --no-deps -r "${PROJECT_ROOT}/requirements/mediapipe.lock.txt"

log "Installing pinned third-party source and checkpoints"
setup_arguments=(
  --runtime "${RUNTIME_ROOT}"
  --project "${PROJECT_ROOT}"
  --python "${VENV}/bin/python"
)
[[ ${OFFLINE} -eq 1 ]] && setup_arguments+=(--offline)
[[ ${WITHOUT_MODELS} -eq 1 ]] && setup_arguments+=(--without-models)
run "${VENV}/bin/python" "${PROJECT_ROOT}/scripts/setup_runtime.py" \
  "${setup_arguments[@]}"

log "Validating the installed runtime"
verify_arguments=(--runtime "${RUNTIME_ROOT}" --project "${PROJECT_ROOT}")
[[ ${WITHOUT_MODELS} -eq 1 ]] && verify_arguments+=(--without-models)
if [[ -n "${RENDER_DEVICE}" ]]; then
  verify_arguments+=(--require-gpu --render-device "${RENDER_DEVICE}")
elif command -v nvidia-smi >/dev/null 2>&1; then
  verify_arguments+=(--require-gpu)
fi
run "${VENV}/bin/python" "${PROJECT_ROOT}/scripts/verify_environment.py" \
  "${verify_arguments[@]}"

log "Setup complete"
echo "Runtime: ${RUNTIME_ROOT}"
echo "Activate: source ${PROJECT_ROOT}/activate.sh"

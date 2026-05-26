#!/usr/bin/env bash
set -Eeuo pipefail

# Provision a GV3 network worker on a fresh AWS Ubuntu machine.
#
# Typical remote use:
#   ssh aws-gpu-Beta 'bash -s' < setup_aws_worker.sh
#
# Useful overrides:
#   REPO_DIR=/home/ubuntu/vidur
#   GIT_REPO_URL=git@github.com:Shaz-hash/vidur.git
#   GIT_BRANCH=gv3-ssh-network
#   BUILD_NATIVE=0
#   RUN_PREFILL_CALIBRATION=1
#   FORCE_PREFILL=0
#   TORCH_INSTALL_COMMAND="/home/ubuntu/vidur/.venv/bin/python -m pip install torch"

REPO_DIR="${REPO_DIR:-/home/ubuntu/vidur}"
GIT_REPO_URL="${GIT_REPO_URL:-git@github.com:Shaz-hash/vidur.git}"
GIT_BRANCH="${GIT_BRANCH:-gv3-ssh-network}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
INSTALL_SYSTEM_PACKAGES="${INSTALL_SYSTEM_PACKAGES:-1}"
SKIP_REPO_SYNC="${SKIP_REPO_SYNC:-0}"
INSTALL_PYTHON_DEPS="${INSTALL_PYTHON_DEPS:-1}"
INSTALL_PROJECT_EDITABLE="${INSTALL_PROJECT_EDITABLE:-0}"
BUILD_NATIVE="${BUILD_NATIVE:-0}"
RUN_PREFILL_CALIBRATION="${RUN_PREFILL_CALIBRATION:-1}"
FORCE_PREFILL="${FORCE_PREFILL:-0}"
PREFILL_STEP="${PREFILL_STEP:-128}"
PREFILL_MAX_TOKENS="${PREFILL_MAX_TOKENS:-4096}"
PREFILL_MODEL_NAME="${PREFILL_MODEL_NAME:-meta-llama/Meta-Llama-3-8B}"
PREFILL_DEVICE="${PREFILL_DEVICE:-a100}"
PREFILL_NETWORK_DEVICE="${PREFILL_NETWORK_DEVICE:-a100_dgx}"
PREFILL_CACHE_DIR="${PREFILL_CACHE_DIR:-${REPO_DIR}/cache}"
NEW_PREFILL_PROFILE="${NEW_PREFILL_PROFILE:-${REPO_DIR}/simulator_output/new_prefill_profile.csv}"
FINAL_PREFILL_PROFILE="${FINAL_PREFILL_PROFILE:-${REPO_DIR}/simulator_output/prefill_profile.csv}"
SETUP_LOG_DIR="${SETUP_LOG_DIR:-${REPO_DIR}/simulator_output/Game_Version3/machine_setup_logs}"

log() {
  printf '[gv3-worker-setup] %s\n' "$*"
}

die() {
  printf '[gv3-worker-setup] ERROR: %s\n' "$*" >&2
  exit 1
}

run() {
  log "+ $*"
  "$@"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

install_system_packages() {
  if [[ "${INSTALL_SYSTEM_PACKAGES}" != "1" ]]; then
    log "Skipping system package install."
    return
  fi
  if ! have_cmd apt-get; then
    log "apt-get not available; skipping system package install."
    return
  fi
  if ! have_cmd sudo && [[ "${EUID}" -ne 0 ]]; then
    log "sudo not available and not root; skipping system package install."
    return
  fi

  local sudo_cmd=()
  if [[ "${EUID}" -ne 0 ]]; then
    sudo_cmd=(sudo)
  fi

  run "${sudo_cmd[@]}" apt-get update
  run "${sudo_cmd[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    ca-certificates \
    cmake \
    git \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    rsync
}

clone_or_update_repo() {
  if [[ "${SKIP_REPO_SYNC}" == "1" ]]; then
    [[ -d "${REPO_DIR}" ]] || die "SKIP_REPO_SYNC=1 but REPO_DIR does not exist: ${REPO_DIR}"
    log "Skipping repo clone/update; using existing repo: ${REPO_DIR}"
    return
  fi

  local parent_dir
  parent_dir="$(dirname "${REPO_DIR}")"
  run mkdir -p "${parent_dir}"

  if [[ -d "${REPO_DIR}/.git" ]]; then
    log "Repo exists: ${REPO_DIR}"
    run git -C "${REPO_DIR}" fetch origin "${GIT_BRANCH}"
    run git -C "${REPO_DIR}" checkout "${GIT_BRANCH}"
    run git -C "${REPO_DIR}" pull --ff-only origin "${GIT_BRANCH}"
    return
  fi

  if [[ -e "${REPO_DIR}" ]] && [[ -n "$(find "${REPO_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || true)" ]]; then
    die "REPO_DIR exists but is not a git repo and is not empty: ${REPO_DIR}"
  fi

  run git clone --branch "${GIT_BRANCH}" "${GIT_REPO_URL}" "${REPO_DIR}"
}

create_venv() {
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    run "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
  run "${VENV_DIR}/bin/python" -m pip install --upgrade pip "setuptools<82" wheel
}

install_python_deps() {
  if [[ "${INSTALL_PYTHON_DEPS}" != "1" ]]; then
    log "Skipping Python dependency install."
    return
  fi

  local py="${VENV_DIR}/bin/python"
  if [[ "${INSTALL_PROJECT_EDITABLE}" == "1" ]]; then
    run "${py}" -m pip install -e "${REPO_DIR}"
  else
    log "Skipping editable project install; worker runs with PYTHONPATH=${REPO_DIR}."
  fi

  run "${py}" -m pip install \
    ddsketch==3.0.1 \
    fasteners==0.17.3 \
    kaleido==0.2.1 \
    matplotlib==3.9.0 \
    numpy==1.26.4 \
    paretoset==1.2.3 \
    plotly-express==0.4.1 \
    pyyaml==6.0.1 \
    randomname==0.2.1 \
    ray==2.31.0 \
    scikit-learn==1.5.0 \
    seaborn==0.13.2 \
    snakeviz==2.2.0 \
    wandb==0.16.6

  if ! "${py}" -c "import torch" >/dev/null 2>&1; then
    local torch_cmd="${TORCH_INSTALL_COMMAND:-${py} -m pip install torch}"
    log "+ ${torch_cmd}"
    bash -lc "${torch_cmd}"
  else
    log "torch already importable."
  fi

  run "${py}" -m pip install pybind11
}

validate_python_dependencies() {
  local py="${VENV_DIR}/bin/python"
  PYTHONPATH="${REPO_DIR}" "${py}" - <<'PY'
import importlib
import platform
import sys

required = ("torch", "numpy", "sklearn", "ray", "yaml", "fasteners")
print("python_version", sys.version.replace("\n", " "))
print("platform_machine", platform.machine())
if sys.version_info < (3, 10):
    raise SystemExit("Python >=3.10 is required")
for name in required:
    mod = importlib.import_module(name)
    version = getattr(mod, "__version__", "unknown")
    print(f"dependency_ok {name} {version}")
PY
}

build_native_module() {
  if [[ "${BUILD_NATIVE}" != "1" ]]; then
    log "Skipping GV3 native module build."
    return
  fi

  local py="${VENV_DIR}/bin/python"
  local native_dir="${REPO_DIR}/vidur/Game_Version3/native"
  local build_dir="${native_dir}/build_aws"
  [[ -d "${native_dir}" ]] || die "GV3 native directory not found: ${native_dir}"

  local pybind11_dir
  pybind11_dir="$("${py}" -m pybind11 --cmakedir)"

  run cmake -S "${native_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="${py}" \
    -Dpybind11_DIR="${pybind11_dir}"
  run cmake --build "${build_dir}" -j "$(nproc)"

  PYTHONPATH="${REPO_DIR}" "${py}" - <<'PY'
import importlib
mod = importlib.import_module("vidur.mcts.mcts_native_gv2")
print("native_import_ok", mod.__name__)
PY
}

run_prefill_calibration() {
  if [[ "${RUN_PREFILL_CALIBRATION}" != "1" ]]; then
    log "Skipping prefill calibration."
    return
  fi

  run mkdir -p "$(dirname "${NEW_PREFILL_PROFILE}")" "${PREFILL_CACHE_DIR}"

  local cache_file_count
  cache_file_count="$(find "${PREFILL_CACHE_DIR}" -type f 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${FORCE_PREFILL}" != "1" ]] && [[ -s "${FINAL_PREFILL_PROFILE}" ]] && [[ "${cache_file_count}" != "0" ]]; then
    log "Prefill profile and cache already exist; skipping calibration: ${FINAL_PREFILL_PROFILE}"
    return
  fi

  local py="${VENV_DIR}/bin/python"

  run env PYTHONPATH="${REPO_DIR}" "${py}" -m vidur.mcts.prefill_calibrator \
    --output "${NEW_PREFILL_PROFILE}" \
    --step "${PREFILL_STEP}" \
    --max_tokens "${PREFILL_MAX_TOKENS}" \
    --replica_config_model_name "${PREFILL_MODEL_NAME}" \
    --replica_config_device "${PREFILL_DEVICE}" \
    --replica_config_network_device "${PREFILL_NETWORK_DEVICE}" \
    --cluster_config_num_replicas 1 \
    --replica_config_tensor_parallel_size 1 \
    --replica_config_num_pipeline_stages 1 \
    --global_scheduler_config_type round_robin \
    --replica_scheduler_config_type vllm_v1 \
    --vllm_v1_scheduler_config_batch_size_cap 512 \
    --execution_time_predictor_config_type random_forest \
    --random_forest_execution_time_predictor_config_prediction_max_tokens_per_request 8192 \
    --random_forest_execution_time_predictor_config_prediction_max_batch_size 256 \
    --random_forest_execution_time_predictor_config_prediction_max_prefill_chunk_size 4096 \
    --random_forest_execution_time_predictor_config_cache_dir "${PREFILL_CACHE_DIR}" \
    --no-snapshot_rng_state

  [[ -s "${NEW_PREFILL_PROFILE}" ]] || die "calibrator did not create profile: ${NEW_PREFILL_PROFILE}"
  run cp "${NEW_PREFILL_PROFILE}" "${FINAL_PREFILL_PROFILE}"
}

validate_worker() {
  local py="${VENV_DIR}/bin/python"
  PYTHONPATH="${REPO_DIR}" "${py}" - <<'PY'
import platform
import torch
from vidur.Game_Version3.Network.client.run_task import main as client_main
from vidur.Game_Version3.Network.network_config import DEFAULT_NETWORK_CONFIG
from vidur.Game_Version3.config import DEFAULT_MULTIPROCESS_TRAINING_CONFIG

print("worker_validation_ok")
print("machine", platform.machine())
print("python_pipeline_environment", DEFAULT_MULTIPROCESS_TRAINING_CONFIG.environment_lang)
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("client_entrypoint", bool(client_main))
print("default_worker_cpu_fraction", DEFAULT_NETWORK_CONFIG.task.worker_cpu_fraction)
PY

  [[ -s "${FINAL_PREFILL_PROFILE}" ]] || die "missing final prefill profile: ${FINAL_PREFILL_PROFILE}"
  log "Final prefill profile: ${FINAL_PREFILL_PROFILE}"
  log "Prefill cache dir: ${PREFILL_CACHE_DIR}"
}

main() {
  install_system_packages
  clone_or_update_repo

  run mkdir -p "${SETUP_LOG_DIR}"
  local log_file="${SETUP_LOG_DIR}/setup_$(date -u +%Y%m%dT%H%M%SZ).log"
  exec > >(tee -a "${log_file}") 2>&1
  log "Logging to ${log_file}"
  log "repo=${REPO_DIR} branch=${GIT_BRANCH}"

  create_venv
  install_python_deps
  build_native_module
  validate_python_dependencies
  run_prefill_calibration
  validate_worker
  log "AWS GV3 worker setup complete."
}

main "$@"

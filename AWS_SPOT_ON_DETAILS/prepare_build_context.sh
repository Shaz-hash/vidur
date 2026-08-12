#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
CONTEXT_DIR="${SCRIPT_DIR}/build_context"
CACHE_SOURCE="${AGZ_EXECUTION_CACHE_SOURCE:-${WORKSPACE_ROOT}/vidur/cache}"

PREFILL_SHA="47f2ed85aca4a5ec47eacae8425d38284e76759b75f5c069852d393ecc62329b"
DECODE_SHA="b14044faaa5f9f5fea1b159f5bd031ce8538b30b59eb8acd32cd9625093c4d2a"
SPOT_RUNTIME_OVERLAY=(
    "vidur/AlphaGoZero/adaptive_rollout.py"
    "vidur/AlphaGoZero/agz_train_eval_promote.py"
    "vidur/AlphaGoZero/bootstrap_untrained_dnn_v100.py"
    "vidur/AlphaGoZero/config.py"
    "vidur/AlphaGoZero/cpp_selfplay_runner.py"
    "vidur/AlphaGoZero/deploy.py"
    "vidur/AlphaGoZero/distributed_eval.py"
    "vidur/AlphaGoZero/dnn_models.py"
    "vidur/AlphaGoZero/durable_transfer.py"
    "vidur/AlphaGoZero/indexed_replay.py"
    "vidur/AlphaGoZero/spot_distributed_eval.py"
    "vidur/AlphaGoZero/spot_work_cli.py"
    "vidur/AlphaGoZero/spot_work_protocol.py"
    "vidur/AlphaGoZero/spot_worker_daemon.py"
    "vidur/AlphaGoZero/worker_daemon.py"
    "vidur/AlphaGoZero/xl_coordinator.py"
    "vidur/bellman_v4_adv/arena_mcts_value_runnerCPP.py"
    "vidur/Game_Version3/alphaZero.py"
    "vidur/Game_Version3/config.py"
    "vidur/Game_Version3/DNN/native_selfplay.py"
    "vidur/Game_Version3/mcts_value_prior.py"
    "vidur/Game_Version3/multiProcessUtils.py"
    "vidur/AlphaGoZero/test_and_analysis/test_dnn_mcts_parity.py"
    "vidur/AlphaGoZero/test_and_analysis/test_dnn_native_parity.py"
    "vidur/AlphaGoZero/test_and_analysis/test_indexed_replay.py"
    "vidur/AlphaGoZero/test_and_analysis/test_markov_policy_dnn.py"
    "vidur/AlphaGoZero/test_and_analysis/test_policy_metrics_parallel.py"
    "vidur/AlphaGoZero/test_and_analysis/test_spot_work_protocol.py"
    "vidur/AlphaGoZero/test_and_analysis/test_spot_worker_dispatch.py"
    "vidur/Game_Version3_Cpp/CMakeLists.txt"
    "vidur/Game_Version3_Cpp/include/agz_dense_dnn.hpp"
    "vidur/Game_Version3_Cpp/include/gv2_cross_game_batcher.hpp"
    "vidur/Game_Version3_Cpp/include/gv2_mcts_dnn.hpp"
    "vidur/Game_Version3_Cpp/include/gv2_types.hpp"
    "vidur/Game_Version3_Cpp/include/new_features_226_inference.hpp"
    "vidur/Game_Version3_Cpp/src/agz_dense_dnn.cpp"
    "vidur/Game_Version3_Cpp/src/gv2_cross_game_batcher.cpp"
    "vidur/Game_Version3_Cpp/src/gv2_mcts_dnn.cpp"
    "vidur/Game_Version3_Cpp/src/new_features_226_inference.cpp"
    "vidur/Game_Version3_Cpp/src/pybind_module.cpp"
)

if [[ ! -d "${CACHE_SOURCE}" ]]; then
    echo "missing execution cache: ${CACHE_SOURCE}" >&2
    exit 2
fi

rm -rf "${CONTEXT_DIR}"
mkdir -p \
    "${CONTEXT_DIR}/repo" \
    "${CONTEXT_DIR}/execution_cache" \
    "${CONTEXT_DIR}/runtime_profiles"

if git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "${REPO_ROOT}" archive --format=tar HEAD \
        README.md pyproject.toml uv.lock data vidur \
        | tar -xf - -C "${CONTEXT_DIR}/repo"
    GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
    SOURCE_DATE_EPOCH="$(git -C "${REPO_ROOT}" show -s --format=%ct HEAD)"
else
    # Production coordinators receive a source snapshot without .git. Copy
    # only package/build inputs; simulator_output and replay data stay outside
    # the context and remain mounted from the coordinator EBS volume.
    tar -C "${REPO_ROOT}" -cf - README.md pyproject.toml uv.lock data vidur \
        | tar -xf - -C "${CONTEXT_DIR}/repo"
    GIT_SHA="deployment-snapshot"
    SOURCE_DATE_EPOCH="$(date -u +%s)"
fi

# Keep the image source narrow and reproducible even when the worktree is dirty.
for relative_path in "${SPOT_RUNTIME_OVERLAY[@]}"; do
    test -f "${REPO_ROOT}/${relative_path}"
    mkdir -p "${CONTEXT_DIR}/repo/$(dirname "${relative_path}")"
    cp "${REPO_ROOT}/${relative_path}" "${CONTEXT_DIR}/repo/${relative_path}"
done

rsync -a --delete "${CACHE_SOURCE%/}/" "${CONTEXT_DIR}/execution_cache/"
cp "${REPO_ROOT}/simulator_output/prefill_profile.csv" \
    "${CONTEXT_DIR}/runtime_profiles/prefill_profile.csv"
cp "${REPO_ROOT}/simulator_output/decode_profile.csv" \
    "${CONTEXT_DIR}/runtime_profiles/decode_profile.csv"
cp "${SCRIPT_DIR}/Dockerfile.worker" "${CONTEXT_DIR}/Dockerfile"
cp "${SCRIPT_DIR}/requirements-worker.txt" "${CONTEXT_DIR}/requirements-worker.txt"
cp "${SCRIPT_DIR}/worker_entrypoint.sh" "${CONTEXT_DIR}/worker_entrypoint.sh"
cp "${SCRIPT_DIR}/container_smoke.py" "${CONTEXT_DIR}/container_smoke.py"

printf '%s  %s\n' "${PREFILL_SHA}" \
    "${CONTEXT_DIR}/runtime_profiles/prefill_profile.csv" | sha256sum -c -
printf '%s  %s\n' "${DECODE_SHA}" \
    "${CONTEXT_DIR}/runtime_profiles/decode_profile.csv" | sha256sum -c -

OVERLAY_SHA="$(
    for relative_path in "${SPOT_RUNTIME_OVERLAY[@]}"; do
        sha256sum "${REPO_ROOT}/${relative_path}"
    done | sha256sum | awk '{print $1}'
)"
cat > "${CONTEXT_DIR}/build-metadata.env" <<EOF
VIDUR_GIT_SHA=${GIT_SHA}
SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}
PREFILL_PROFILE_SHA256=${PREFILL_SHA}
DECODE_PROFILE_SHA256=${DECODE_SHA}
SPOT_RUNTIME_OVERLAY_SHA256=${OVERLAY_SHA}
EOF

echo "prepared=${CONTEXT_DIR}"
du -sh "${CONTEXT_DIR}"
cat "${CONTEXT_DIR}/build-metadata.env"

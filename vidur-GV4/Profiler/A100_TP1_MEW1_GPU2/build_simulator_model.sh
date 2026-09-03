#!/usr/bin/env bash
set -euo pipefail

readonly PROFILE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "$PROFILE_ROOT/../../.." && pwd)"
readonly PYTHON="${VIDUR_PROFILE_PYTHON:-$REPO_ROOT/.venv/bin/python}"
readonly JOBS="${VIDUR_PROFILE_TRAIN_JOBS:-32}"
readonly MODEL="meta-llama/Meta-Llama-3-8B"
readonly RAW_COMPUTE_ROOT="$REPO_ROOT/data/profiling/compute/a100_mew1_gpu2/$MODEL"
readonly COMPUTE_ROOT="$REPO_ROOT/data/profiling/compute/a100_mew1_gpu2_vllm013_calibrated/$MODEL"
readonly ARTIFACTS="$PROFILE_ROOT/artifacts"
readonly CACHE_DIR="$ARTIFACTS/cache"
readonly RAW_CACHE_DIR="$ARTIFACTS/raw_cache"
readonly METRICS_DIR="$ARTIFACTS/build_metrics"
readonly PREFILL_PROFILE="$ARTIFACTS/prefill_profile.csv"
readonly DECODE_PROFILE="$ARTIFACTS/decode_profile.csv"
readonly RAW_PREFILL_PROFILE="$ARTIFACTS/raw_prefill_profile.csv"
readonly RAW_DECODE_PROFILE="$ARTIFACTS/raw_decode_profile.csv"
readonly CALIBRATION_METADATA="$ARTIFACTS/calibration.json"

if [[ ! -x "$PYTHON" ]]; then
    echo "missing Python environment: $PYTHON" >&2
    exit 2
fi
for input_file in "$RAW_COMPUTE_ROOT/mlp.csv" "$RAW_COMPUTE_ROOT/attention.csv"; do
    if [[ ! -s "$input_file" ]]; then
        echo "missing profiling input: $input_file" >&2
        exit 2
    fi
done

mkdir -p "$CACHE_DIR" "$RAW_CACHE_DIR" "$METRICS_DIR" "$ARTIFACTS/validation"

base_args=(
    --replica_config_model_name "$MODEL"
    --replica_config_device a100
    --replica_config_network_device a100_dgx
    --cluster_config_num_replicas 1
    --replica_config_tensor_parallel_size 1
    --replica_config_num_pipeline_stages 1
    --global_scheduler_config_type round_robin
    --replica_scheduler_config_type vllm_v1
    --vllm_v1_scheduler_config_batch_size_cap 512
    --execution_time_predictor_config_type random_forest
    --random_forest_execution_time_predictor_config_prediction_max_tokens_per_request 8192
    --random_forest_execution_time_predictor_config_prediction_max_batch_size 256
    --random_forest_execution_time_predictor_config_prediction_max_prefill_chunk_size 4096
    --random_forest_execution_time_predictor_config_cache_mode use_cache
    --random_forest_execution_time_predictor_config_num_training_job_threads "$JOBS"
    --metrics_config_output_dir "$METRICS_DIR"
    --no-snapshot_rng_state
)

export PYTHONPATH="$REPO_ROOT"

raw_args=(
    "${base_args[@]}"
    --random_forest_execution_time_predictor_config_compute_input_file "$RAW_COMPUTE_ROOT/mlp.csv"
    --random_forest_execution_time_predictor_config_attention_input_file "$RAW_COMPUTE_ROOT/attention.csv"
    --random_forest_execution_time_predictor_config_cache_dir "$RAW_CACHE_DIR"
)

"$PYTHON" -m vidur.Game_Version3.prefill_calibrator \
    --output "$RAW_PREFILL_PROFILE" \
    --step 128 \
    --max_tokens 4096 \
    "${raw_args[@]}"

"$PYTHON" "$PROFILE_ROOT/generate_decode_profile.py" \
    --output "$RAW_DECODE_PROFILE" \
    --tokens 128,256,512,1024,1536,2048,2560,3072,3584,4096 \
    "${raw_args[@]}"

"$PYTHON" "$PROFILE_ROOT/calibrate_operation_profiles.py" \
    --raw-compute-root "$RAW_COMPUTE_ROOT" \
    --output-compute-root "$COMPUTE_ROOT" \
    --raw-prefill-profile "$RAW_PREFILL_PROFILE" \
    --real-profile "$PROFILE_ROOT/raw/real_vllm/prefill_profile_comparison.csv" \
    --output-metadata "$CALIBRATION_METADATA"

common_args=(
    "${base_args[@]}"
    --random_forest_execution_time_predictor_config_compute_input_file "$COMPUTE_ROOT/mlp.csv"
    --random_forest_execution_time_predictor_config_attention_input_file "$COMPUTE_ROOT/attention.csv"
    --random_forest_execution_time_predictor_config_cache_dir "$CACHE_DIR"
)

"$PYTHON" -m vidur.Game_Version3.prefill_calibrator \
    --output "$PREFILL_PROFILE" \
    --step 128 \
    --max_tokens 4096 \
    "${common_args[@]}"

"$PYTHON" "$PROFILE_ROOT/generate_decode_profile.py" \
    --output "$DECODE_PROFILE" \
    --tokens 128,256,512,1024,1536,2048,2560,3072,3584,4096 \
    "${common_args[@]}"

"$PYTHON" "$PROFILE_ROOT/validate_simulator_profile.py" \
    --simulator-profile "$PREFILL_PROFILE" \
    --real-profile "$PROFILE_ROOT/raw/real_vllm/prefill_profile_comparison.csv" \
    --legacy-profile "$REPO_ROOT/simulator_output/prefill_profile.csv" \
    --output-csv "$ARTIFACTS/validation/prefill_comparison.csv" \
    --output-json "$ARTIFACTS/validation/prefill_comparison.json"

"$PYTHON" - "$PROFILE_ROOT" "$RAW_COMPUTE_ROOT" "$COMPUTE_ROOT" "$CACHE_DIR" "$CALIBRATION_METADATA" <<'PY'
import hashlib
import json
import platform
import sys
from pathlib import Path

profile_root = Path(sys.argv[1])
raw_compute_root = Path(sys.argv[2])
compute_root = Path(sys.argv[3])
cache_dir = Path(sys.argv[4])
calibration_metadata = Path(sys.argv[5])

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    "model": "meta-llama/Meta-Llama-3-8B",
    "device": "A100 80GB PCIe on mew1 physical GPU 2",
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "prediction_max_tokens_per_request": 8192,
    "prediction_max_batch_size": 256,
    "prediction_max_prefill_chunk_size": 4096,
    "prefill_step": 128,
    "backend_calibration": json.loads(calibration_metadata.read_text(encoding="utf-8")),
    "python": platform.python_version(),
    "raw_inputs": {
        name: {"path": str(path), "sha256": digest(path)}
        for name, path in {
            "mlp": raw_compute_root / "mlp.csv",
            "attention": raw_compute_root / "attention.csv",
        }.items()
    },
    "calibrated_inputs": {
        name: {"path": str(path), "sha256": digest(path)}
        for name, path in {
            "mlp": compute_root / "mlp.csv",
            "attention": compute_root / "attention.csv",
        }.items()
    },
    "artifacts": {
        "prefill_profile": str(profile_root / "artifacts/prefill_profile.csv"),
        "decode_profile": str(profile_root / "artifacts/decode_profile.csv"),
        "raw_prefill_profile": str(profile_root / "artifacts/raw_prefill_profile.csv"),
        "raw_decode_profile": str(profile_root / "artifacts/raw_decode_profile.csv"),
        "cache": str(cache_dir),
    },
    "cache_file_count": sum(1 for path in cache_dir.rglob("*") if path.is_file()),
}
(profile_root / "artifacts/manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
PY

echo "Simulator model built under $ARTIFACTS"

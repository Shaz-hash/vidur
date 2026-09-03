#!/usr/bin/env bash
set -euo pipefail

readonly HOST="${VIDUR_MEW1_SSH_HOST:-mew1}"
readonly ROOT="${VIDUR_MEW1_ROOT:-/home/shaz/vidur}"
readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly REMOTE_SOURCE="$ROOT/source/vidur-classical-search"
readonly MODEL_SOURCE="$REPO_ROOT/simulator_output/GV3_Agent/Synthetic_Trace_Tester/arena_like_trace_20s/eval_current_xl3_xl4_98_2k_rollout3s/models"

if [[ "$HOST" != "mew1" ]]; then
    echo "refusing to deploy to $HOST; only mew1 is allowed" >&2
    exit 2
fi
case "$ROOT" in
    /home/shaz/*) ;;
    *) echo "VIDUR_MEW1_ROOT must remain under /home/shaz" >&2; exit 2 ;;
esac
if [[ ! -d "$MODEL_SOURCE" ]]; then
    echo "promoted model bundle directory is missing: $MODEL_SOURCE" >&2
    exit 2
fi
for profile in prefill_profile.csv decode_profile.csv; do
    if [[ ! -f "$REPO_ROOT/simulator_output/$profile" ]]; then
        echo "required simulator profile is missing: $profile" >&2
        exit 2
    fi
done

ssh "$HOST" "mkdir -p '$REMOTE_SOURCE/simulator_output' '$ROOT/artifacts/promoted_models' '$ROOT/manifests'"

common_excludes=(
    --exclude='.git/'
    --exclude='.venv/'
    --exclude='__pycache__/'
    --exclude='*.pyc'
    --exclude='*.so'
    --exclude='build/'
    --exclude='build_*/'
    --exclude='.pytest_cache/'
    --exclude='.mypy_cache/'
)

rsync -az --delete "${common_excludes[@]}" \
    "$REPO_ROOT/vidur/" "$HOST:$REMOTE_SOURCE/vidur/"
rsync -az --delete "${common_excludes[@]}" \
    "$REPO_ROOT/vidur_vllm_real_testing/" \
    "$HOST:$REMOTE_SOURCE/vidur_vllm_real_testing/"
rsync -az "$REPO_ROOT/pyproject.toml" "$HOST:$REMOTE_SOURCE/pyproject.toml"
rsync -az \
    "$REPO_ROOT/simulator_output/prefill_profile.csv" \
    "$REPO_ROOT/simulator_output/decode_profile.csv" \
    "$HOST:$REMOTE_SOURCE/simulator_output/"
rsync -az --delete "$MODEL_SOURCE/" "$HOST:$ROOT/artifacts/promoted_models/"

local_commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
local_dirty="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=no 2>/dev/null | wc -l)"
deployed_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
ssh "$HOST" "printf 'git_commit=%s\ntracked_dirty_entries=%s\ndeployed_at_utc=%s\n' '$local_commit' '$local_dirty' '$deployed_at_utc' > '$ROOT/manifests/source.txt'"
ssh "$HOST" "cd '$REMOTE_SOURCE' && find vidur vidur_vllm_real_testing simulator_output -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > '$ROOT/manifests/source.sha256'"

echo "source, profiles, and promoted models deployed to $HOST:$ROOT"

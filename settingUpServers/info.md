# Bellman Classical Worker Setup

## Current AlphaGoZero Cluster Bootstrap

For a fresh Arm64 AlphaGoZero cluster, update
`settingUpServers/agz_clusters.json` and the local SSH aliases, then run:

```bash
cd /home/shazer/Desktop/Research/Vidur/vidur-classical-search
python3 settingUpServers/agz_cluster_setup.py all --cluster exp3 --deep
```

The command is idempotent and performs these stages in order:

1. Verify SSH, Arm64, CPU count, memory, and disk.
2. Install the deployment key and SSH config so each target can pull directly
   from its matching validated reference host.
3. Install the established Ubuntu build/runtime package set, including
   `build-essential`, `cmake`, `pkg-config`, Python tooling, Git, and rsync.
4. Sync current local source while excluding replay/output/runtime artifacts.
5. Clone CPython, the virtual environment, Vidur execution cache, KV cache, and
   `prefill_profile.csv` from the matching EXP2 role.
6. Clone each worker's matching parent-state dataset.
7. Rebuild `Game_Version3_Cpp` in release mode on every target.
8. Validate package imports, native rollout CLI support, profile hash, dataset
   presence, and resource floors.
9. With `--deep`, regenerate the 32-point prefill profile from the copied
   execution cache and require an exact hash match.

Individual stages can be rerun without repeating the full setup:

```bash
python3 settingUpServers/agz_cluster_setup.py sync-code --cluster exp3
python3 settingUpServers/agz_cluster_setup.py install-system --cluster exp3
python3 settingUpServers/agz_cluster_setup.py build-native --cluster exp3
python3 settingUpServers/agz_cluster_setup.py validate --cluster exp3 --deep
```

Before launching a rollout experiment, measure native search under the same
worker concurrency rather than extrapolating a single-process benchmark:

```bash
ssh bellman-classical-exp3-worker-2 'cd /home/ubuntu/vidur-classical-search && \
  .venv/bin/python3 settingUpServers/agz_rollout_concurrency_benchmark.py \
  --model-root simulator_output/GV3_Agent/AlphaGoZero_EXP3_UNTRAINED_.995_500_MCTS_1s_LEAF_RELATIVE_ROLLOUT_DNN/models/Model_Version100 \
  --output-dir /tmp/exp3_rollout_benchmark_t1 --processes 60 \
  --iterations 500 --rollout-count 10 --rollout-parallel-threads 1 \
  --rollout-horizon-sec 1.0 --discount-factor 0.995'
```

The benchmark writes one result per process and a machine-readable
`summary.json` containing throughput, latency percentiles, failures, and peak
combined RSS.

## CPU- and RAM-aware distributed evaluation

Role and SJF evaluation no longer use one fixed process/thread layout. After
self-play is paused, the coordinator samples live `/proc/meminfo` and
`/proc/stat` on XL and all eight workers, intersects that capacity with the
configured CPU set and computes:

`game_slots = min(RAM slots, configured process cap, usable cores / threads per game)`

The configured CPU sets retain cores `0-3` on XL and `0-1` on every worker for
the coordinator, SSH, and OS work; no second reserve is subtracted from those
already reduced sets. Self-play keeps 10 rollout trajectories and one native
rollout thread so training data quality is unchanged. Evaluation uses three
rollout trajectories: two native threads per role-evaluation game and three per
SJF game. All 400 role games and all 50 paired SJF games must fit in one wave;
otherwise the phase fails closed rather than oversubscribing memory or silently
running extra waves. Baseline/candidate games with the same seed and history hop
remain paired on the same host. Model staging is parallel across hosts. Every
`launch_command.json` records the sampled idle fraction, selected and usable
cores, reserve, thread count, and final game slots.

The complete v104 role evaluation took `972.09` seconds (`16m12s`) with two
threads, compared with `1357.83` seconds (`22m38s`) with one thread and
`3849.69` seconds (`64m10s`) with the original 10-trajectory evaluation. All
four 100-game blocks were semantically identical between the one- and
two-thread runs. The complete v103 50-pair SJF evaluation took `637.31` seconds
(`10m37s`) with three threads, compared with `1146.84` seconds (`19m07s`) with
one thread and `2631.7` seconds (`43m52s`) under the original settings. All 50
SJF rows were semantically identical between the one- and three-thread runs.
This satisfies the 20-minute role target and keeps role plus SJF evaluation
under the 30-minute combined target.

The EXP3 runtime contract is stored in
`settingUpServers/exp3_rollout_env.sh`. After the one-command setup and v100
bootstrap, launch the coordinator and workers with:

```bash
cd /home/shazer/Desktop/Research/Vidur/vidur-classical-search
set -a
source settingUpServers/exp3_rollout_env.sh
set +a

.venv/bin/python3 -m vidur.AlphaGoZero.deploy \
  --remote-output-root "$AGZ_REMOTE_OUTPUT_ROOT" \
  launch-xl-ingest --poll-sec 10

.venv/bin/python3 -m vidur.AlphaGoZero.deploy \
  --remote-output-root "$AGZ_REMOTE_OUTPUT_ROOT" \
  launch-worker-large --workers all \
  --run-id agz_exp3_leaf_relative_995_500_rollout1s_v100 \
  --iterations 500 --history-hops 0 --arena-time-limit-sec 5 \
  --game-id-start 1100000000 --max-games 0 --parallel-games 60 \
  --buffer-threshold 4000 --upload-poll-sec 30 --ack-timeout-sec 0 \
  --agz-sample-initial-move-count 20 --parent-root-player-filter any
```

Do not run the launch commands from a shell that has stale AlphaGoZero
variables. The environment file is the source of truth for discount, rollout,
training-gate, replay, and evaluation settings.

The machine/IP/instance mapping is stored in
`settingUpServers/agz_clusters.json`; experiment ownership and runtime knobs
are stored in `vidur/AlphaGoZero/machines.md`. Do not use system Python and do
not copy a native `.so` from another host: copy the validated runtime, then
compile current source locally on every target.


This document records the setup used for the additional `c8g.24xlarge` root-generation workers.

## Hosts

SSH aliases added in `~/.ssh/config`:

```sshconfig
Host bellman-classical-worker-1
    HostName 18.206.191.97
    User ubuntu
    IdentityFile ~/.ssh/shaz-pr1.pem
    IdentitiesOnly yes

Host bellman-classical-worker-2
    HostName 3.94.7.204
    User ubuntu
    IdentityFile ~/.ssh/shaz-pr1.pem
    IdentitiesOnly yes

Host bellman-classical-worker-3
    HostName 54.242.30.220
    User ubuntu
    IdentityFile ~/.ssh/shaz-pr1.pem
    IdentitiesOnly yes

Host bellman-classical-worker-4
    HostName 18.232.106.21
    User ubuntu
    IdentityFile ~/.ssh/shaz-pr1.pem
    IdentitiesOnly yes

Host bellman-classical-worker-5
    HostName 3.90.165.3
    User ubuntu
    IdentityFile ~/.ssh/shaz-pr1.pem
    IdentitiesOnly yes

Host bellman-classical-worker-6
    HostName 54.198.117.83
    User ubuntu
    IdentityFile ~/.ssh/shaz-pr1.pem
    IdentitiesOnly yes

Host bellman-classical-worker-7
    HostName 18.207.165.120
    User ubuntu
    IdentityFile ~/.ssh/shaz-pr1.pem
    IdentitiesOnly yes

Host bellman-classical-worker-8
    HostName 100.54.208.103
    User ubuntu
    IdentityFile ~/.ssh/shaz-pr1.pem
    IdentitiesOnly yes
```

Each machine is Ubuntu 26.04 on `aarch64`, with 96 cores, about 185 GiB RAM, and about 991 GiB root disk.

## Reference Machine

Use `bellman-classical-xl` as the runtime/cache reference:

- Reference repo: `/home/ubuntu/vidur-classical-search`
- Reference Python: `/home/ubuntu/vidur-classical-search/.venv/bin/python3`
- Python runtime target: `/home/ubuntu/.local/share/uv`
- Python version: `3.10.20`
- Package spot-check: `numpy==1.26.4`, `scikit-learn==1.5.0`
- Vidur cache: `/home/ubuntu/vidur-classical-search/cache`
- Profiles:
  - `simulator_output/prefill_profile.csv`
  - `simulator_output/decode_profile.csv`

Do not use system Python on new workers. The current AMI has Python `3.14.4`, which is not the validated runtime for this repo.

## Setup Steps

Create target directories:

```bash
ssh <worker> 'mkdir -p /home/ubuntu/vidur-classical-search /home/ubuntu/.local/share/uv /home/ubuntu/.local/bin /home/ubuntu/vidur-classical-search/simulator_output /home/ubuntu/vidur-classical-search/vidur'
```

Sync repo code from local, excluding heavy/generated artifacts:

```bash
rsync -az --delete \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='cache' \
  --exclude='__pycache__' \
  --include='simulator_output/' \
  --include='simulator_output/prefill_profile.csv' \
  --include='simulator_output/decode_profile.csv' \
  --exclude='simulator_output/***' \
  /home/shazer/Desktop/Research/Vidur/vidur-classical-search/ \
  <worker>:/home/ubuntu/vidur-classical-search/
```

Stage the known-good ARM runtime locally once:

```bash
mkdir -p /tmp/vidur_bellman_ref_env/home_ubuntu_local_share_uv
mkdir -p /tmp/vidur_bellman_ref_env/home_ubuntu_local_bin
mkdir -p /tmp/vidur_bellman_ref_env/repo

rsync -az --delete bellman-classical-xl:/home/ubuntu/.local/share/uv/ /tmp/vidur_bellman_ref_env/home_ubuntu_local_share_uv/
rsync -az --delete bellman-classical-xl:/home/ubuntu/.local/bin/ /tmp/vidur_bellman_ref_env/home_ubuntu_local_bin/
rsync -az --delete bellman-classical-xl:/home/ubuntu/vidur-classical-search/.venv/ /tmp/vidur_bellman_ref_env/repo/.venv/
```

Fan out the staged runtime and venv:

```bash
rsync -az --delete /tmp/vidur_bellman_ref_env/home_ubuntu_local_share_uv/ <worker>:/home/ubuntu/.local/share/uv/
rsync -az --delete /tmp/vidur_bellman_ref_env/home_ubuntu_local_bin/ <worker>:/home/ubuntu/.local/bin/
rsync -az --delete /tmp/vidur_bellman_ref_env/repo/.venv/ <worker>:/home/ubuntu/vidur-classical-search/.venv/
```

Sync the Vidur execution cache and small KV cache:

```bash
rsync -az --delete /home/shazer/Desktop/Research/Vidur/vidur/cache/ <worker>:/home/ubuntu/vidur-classical-search/cache/
rsync -az --delete /home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/kv_cache/ <worker>:/home/ubuntu/vidur-classical-search/vidur/kv_cache/
```

## Validation

Runtime/package check:

```bash
ssh <worker> 'cd /home/ubuntu/vidur-classical-search && .venv/bin/python3 -c "import sys,numpy,sklearn; print(sys.version.split()[0], numpy.__version__, sklearn.__version__)"'
```

Expected:

```text
3.10.20 1.26.4 1.5.0
```

Profile hash check:

```bash
ssh <worker> 'cd /home/ubuntu/vidur-classical-search && sha256sum simulator_output/prefill_profile.csv simulator_output/decode_profile.csv'
```

Expected:

```text
47f2ed85aca4a5ec47eacae8425d38284e76759b75f5c069852d393ecc62329b  simulator_output/prefill_profile.csv
b14044faaa5f9f5fea1b159f5bd031ce8538b30b59eb8acd32cd9625093c4d2a  simulator_output/decode_profile.csv
```

Cache/package size spot-check:

```bash
ssh <worker> 'cd /home/ubuntu/vidur-classical-search && du -sh .venv cache vidur/kv_cache && df -h /home/ubuntu'
```

Expected approximate sizes:

- `.venv`: `5.3G`
- `cache`: `3.5G`
- `vidur/kv_cache`: small; verify file hashes if needed because filesystem block-size reporting can differ.

Compile check:

```bash
ssh <worker> 'cd /home/ubuntu/vidur-classical-search && .venv/bin/python3 -m py_compile vidur/Game_Version3/ModelSearchBed/analysis_testing/rootGeneration.py vidur/Game_Version3/ModelSearchBed/root_storage.py vidur/bellman_v4_adv_2000k_multiprocess/multi_server_root_generation.py vidur/tests/game_engine_tests.py'
```

Root-generation smoke check:

```bash
/home/shazer/Desktop/Research/Vidur/vidur-classical-search/.venv/bin/python3 \
  /home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/bellman_v4_adv_2000k_multiprocess/multi_server_root_generation.py \
  smoke \
  --smoke-host <worker> \
  --smoke-label setup_smoke_<worker> \
  --smoke-roots 10 \
  --smoke-max-candidates 5000 \
  --smoke-num-processes 4 \
  --smoke-worker-roots-per-task 5 \
  --no-sync \
  --force
```

Expected summary shape:

```json
{
  "stats": {
    "roots_stored": 10,
    "nonzero_roots_stored": 6,
    "zero_roots_stored": 4
  },
  "nonzero_ratio": 0.6
}
```

This was verified on all four additional workers. Worker 4 needed `--smoke-max-candidates 5000`; the smaller `1000` candidate smoke cap ended early at 9 roots because the candidate budget was exhausted, not because setup failed.

Prefill profile parity check with the current GV3 A100 cached execution config:

```bash
ssh <worker> 'cd /home/ubuntu/vidur-classical-search && .venv/bin/python3 -m vidur.Game_Version3.prefill_calibrator --output /tmp/prefill_profile_check_a100.csv --step 128 --max_tokens 4096 --replica_config_model_name meta-llama/Meta-Llama-3-8B --replica_config_device a100 --replica_config_network_device a100_dgx --cluster_config_num_replicas 1 --replica_config_tensor_parallel_size 1 --replica_config_num_pipeline_stages 1 --global_scheduler_config_type round_robin --replica_scheduler_config_type vllm_v1 --vllm_v1_scheduler_config_batch_size_cap 512 --execution_time_predictor_config_type random_forest --random_forest_execution_time_predictor_config_prediction_max_tokens_per_request 8192 --random_forest_execution_time_predictor_config_prediction_max_batch_size 256 --random_forest_execution_time_predictor_config_prediction_max_prefill_chunk_size 4096 --random_forest_execution_time_predictor_config_cache_dir /home/ubuntu/vidur-classical-search/cache --random_forest_execution_time_predictor_config_cache_mode require_cache --random_forest_execution_time_predictor_config_num_training_job_threads 1 --no-snapshot_rng_state > /tmp/prefill_profile_check_a100.log 2>&1 && sha256sum /tmp/prefill_profile_check_a100.csv simulator_output/prefill_profile.csv'
```

Expected: both hashes are identical:

```text
47f2ed85aca4a5ec47eacae8425d38284e76759b75f5c069852d393ecc62329b
```

Do not validate with the older H100 example command unless intentionally testing H100 behavior. It produces a different, faster profile and does not match the current GV3 A100 profile.

## Root Generation Coordinator

The multi-server coordinator is:

```text
/home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/bellman_v4_adv_2000k_multiprocess/multi_server_root_generation.py
```

Its default host list now contains:

```text
bellman-classical
bellman-classical-xl
bellman-classical-worker-1
bellman-classical-worker-2
bellman-classical-worker-3
bellman-classical-worker-4
bellman-classical-worker-5
bellman-classical-worker-6
bellman-classical-worker-7
bellman-classical-worker-8
```

Use this script to sync, launch, status, stop, and collect root-generation jobs. It delegates to `vidur/Game_Version3/ModelSearchBed/analysis_testing/rootGeneration.py` on each remote machine.

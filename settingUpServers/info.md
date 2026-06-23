# Bellman Classical Worker Setup

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

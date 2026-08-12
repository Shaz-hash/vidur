# AWS Spot AlphaGoZero Worker Image

This directory packages commit `3892281` of the EXP3 AlphaGoZero runtime into
a worker image. It intentionally does not contain replay, model versions,
parent-state datasets, SSH credentials, or worker output.

## Image Contract

- Application path: `/home/ubuntu/vidur-classical-search`
- Python: `3.12.13`
- Architectures: `linux/amd64` and `linux/arm64`
- Native module: compiled inside the target architecture image
- Included runtime data:
  - validated Vidur execution cache
  - `prefill_profile.csv`
  - `decode_profile.csv`
- External mounts:
  - experiment models and `current_model.json`
  - worker output/ready shards
  - parent-state dataset
  - SSH key/config for controller requests and direct XL transfer

The native build replaces `-march=native` only inside the image:

- ARM64: `-march=armv8.2-a`, compatible with Graviton 2/3/4
- AMD64: `-march=x86-64-v3`

Game and MCTS semantics are unchanged.

## Build Locally

```bash
cd /home/shazer/Desktop/Research/Vidur/vidur-classical-search
AWS_SPOT_ON_DETAILS/prepare_build_context.sh
AWS_SPOT_ON_DETAILS/build_image.sh
docker run --rm vidur-agz-exp3-worker:rollout-modes-3892281 smoke
```

The local host needs Docker Engine. Cross-building ARM64 also needs buildx and
QEMU/binfmt support.

## Export ARM64 For The Temporary Machine

```bash
AWS_SPOT_ON_DETAILS/export_arm64_image.sh
```

This produces:

```text
AWS_SPOT_ON_DETAILS/artifacts/
  vidur-agz-exp3-worker-rollout-modes-3892281-linux-arm64.tar
  vidur-agz-exp3-worker-rollout-modes-3892281-linux-arm64.tar.sha256
```

Transfer and load it:

```bash
rsync -az --progress AWS_SPOT_ON_DETAILS/artifacts/*.tar* <host>:/tmp/
ssh <host> 'cd /tmp && sha256sum -c vidur-agz-exp3-worker-rollout-modes-3892281-linux-arm64.tar.sha256'
ssh <host> 'docker load -i /tmp/vidur-agz-exp3-worker-rollout-modes-3892281-linux-arm64.tar'
ssh <host> 'docker run --rm vidur-agz-exp3-worker:rollout-modes-3892281 smoke'
```

## Pull Scheduling

`spot-worker` asks the XL controller for work before every bounded wave:

1. Queued evaluation games have priority.
2. An active self-play wave receives an explicit preemption response on its
   lease heartbeat, flushes completed rows, and terminates unfinished games.
3. While an evaluation batch is active, a worker either receives leased eval
   games or waits. It never starts self-play in the remaining eval window.
4. With no active evaluation, it receives self-play using the exact current
   promoted controller and adversary versions.

Assignments carry model versions, artifact directories, file sizes, and
SHA-256 hashes. Workers verify those exact files before launching native games.
Eval assignments use renewable lease tokens, heartbeats, retries, checksummed
result upload, and stale-token rejection. Self-play uses the existing shard
upload and coordinator acknowledgement path.

Enable pull-based arena evaluation on the coordinator with:

```bash
export AGZ_SPOT_PULL_EVAL_ENABLED=1
export AGZ_SPOT_PULL_EVAL_TIMEOUT_SEC=7200
export AGZ_SPOT_PULL_EVAL_POLL_SEC=2
```

Leaving `AGZ_SPOT_PULL_EVAL_ENABLED=0` preserves the fixed-host evaluation
path.

## Resource-Aware Fleet Scheduling

XL owns two experiment-wide concurrency targets:

```bash
export AGZ_SPOT_SELFPLAY_TOTAL_PARALLEL_GAMES=500
export AGZ_SPOT_EVAL_TOTAL_PARALLEL_GAMES=280
export AGZ_SPOT_SELFPLAY_LEASE_SEC=300
export AGZ_SPOT_WORKER_HEARTBEAT_TIMEOUT_SEC=300
```

Starting `xl_coordinator` with those variables persists the limits in
`spot_work/control/scheduler_config.json`. They can also be configured directly:

```bash
python -m vidur.AlphaGoZero.spot_work_cli \
  --root /path/to/experiment \
  configure \
  --selfplay-total-parallel-games 500 \
  --eval-total-parallel-games 280 \
  --selfplay-lease-sec 300 \
  --worker-heartbeat-timeout-sec 300
```

The same scheduler file is also the authoritative source for every self-play
game/search parameter. It includes MCTS iterations, discount, horizon, PUCT/UCT,
Dirichlet settings, initial-move sampling, rollout mode/count/threads,
temperatures, seeds, and parent-state selection. The coordinator fingerprints
the complete schema; workers reject missing, unknown, or hash-mismatched fields
and overwrite their image defaults before launching native self-play.

Evaluation is controller-owned separately. XL queues the complete per-game
arena command and its SHA-256 fingerprint. Workers verify that fingerprint
before execution, so evaluation iterations, rollout settings, models, seeds,
and game settings cannot silently fall back to worker defaults.

Before every wave, each worker reports:

- CPUs visible through its process affinity and cgroup CPU quota.
- CPUs reserved for the uploader, SSH, and operating-system overhead.
- CPU threads required by one game.
- Available host/cgroup memory.
- Configured memory required by one game, one GiB by default.

XL independently computes:

```text
worker slots = min(
    worker safety cap,
    floor((visible CPUs - reserved CPUs) / CPU threads per game),
    floor(available RAM / RAM per game),
    unallocated experiment-wide slots
)
```

Automatic CPU reservation is `max(2, ceil(cpus / 16))`, bounded so a
single-core machine can still run one game. This gives 14 game slots on a
16-core machine, 30 on a 32-core machine, and 90 on a 96-core machine before
applying the RAM and global limits. Set `--resource-reserve-cpus` to override
it and `--resource-memory-gib-per-game` if measured per-game memory differs
from one GiB.

Self-play grants and eval tasks use renewable leases. Workers heartbeat every
60 seconds, while XL enforces `worker_heartbeat_timeout_sec=300` as a hard
upper bound on both initial leases and every renewal. A worker cannot extend a
reservation past five minutes from its last accepted heartbeat, even if an old
image or operator requests 3,600 or 7,200 seconds; values above 300 are clamped
in the protocol. The next worker pull or scheduler-status check transactionally
removes the stale self-play reservation or requeues the eval task with a new
token. The replacement worker polls every two to five seconds.

This bound covers control-plane recovery, not EC2 provisioning time. Auto
Scaling Capacity Rebalancing and ASG replacement determine how quickly a new
instance boots. Accepted shards, replay partitions, model bundles, completed
eval results, and CSVs live on XL and survive worker loss. An interrupted
worker can lose only its active games and the partial local shard that had not
yet reached the upload threshold. A late eval result is rejected by its stale
lease token; duplicate self-play computation cannot overwrite accepted data.

Inspect the live allocation with:

```bash
python -m vidur.AlphaGoZero.spot_work_cli \
  --root /path/to/experiment scheduler-status
```

## Spot Worker Runtime

Keep existing EXP3 absolute paths so model manifests do not need rewriting:

```bash
docker run --rm \
  --name agz-spot-worker \
  --network host \
  -v /home/ubuntu/vidur-classical-search/simulator_output:/home/ubuntu/vidur-classical-search/simulator_output \
  -v /home/ubuntu/.ssh:/home/ubuntu/.ssh \
  vidur-agz-exp3-worker:resource-aware-v1-20260731 \
  spot-worker \
  --worker-id spot-test-001 \
  --output-root /home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/spot-test/worker_buffer \
  --xl-host bellman-classical-exp3-xl \
  --xl-output-root /home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/AlphaGoZero_EXP3 \
  --controller-repo /home/ubuntu/vidur-classical-search \
  --controller-python /home/ubuntu/vidur-classical-search/.venv/bin/python3 \
  --parallel-games 0 \
  --buffer-threshold 4000 \
  --assignment-lease-sec 300 \
  --heartbeat-sec 10 \
  --resource-reserve-cpus -1 \
  --resource-memory-gib-per-game 1
```

The parent-state dataset must also be mounted at the same absolute path when
`--parent-dataset-dir` is used.

The SSH mount is writable because OpenSSH may update `known_hosts`. The
controller host alias, repository, interpreter, and experiment root must
resolve inside the container. This direct-SSH design does not require S3 or
ECS; a fleet manager only needs to maintain the desired container count.

## Entrypoint Modes

- `smoke`: validate packages, profiles, cache, native module, and worker CLIs.
- `worker`: run `vidur.AlphaGoZero.worker_daemon`.
- `spot-worker`: pull eval or self-play work from the XL controller.
- `arena`: run one or more native evaluation games.
- `shell`: open a diagnostic shell.

See [pipeline_process_map.md](pipeline_process_map.md) for coordinator training
process counts, worker game concurrency, and the evaluation-preemption
lifecycle.

## EC2 Launch User Data

For the first fleet, each worker receives the existing coordinator SSH private
key through launch-template user data. Secrets Manager is not required. The
key is base64-encoded in the rendered script, decoded during boot, written to
`/var/lib/vidur-agz/ssh/id_ed25519` with mode `0600`, and excluded from the
worker container environment. All workers launched from that template share
the same coordinator credential.

Render user data locally:

```bash
export AGZ_XL_HOST=3.92.29.240
export AGZ_XL_OUTPUT_ROOT=/home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/<experiment>
export AGZ_SSH_PRIVATE_KEY_PATH=$HOME/.ssh/shaz-pr1.pem
export AGZ_USER_DATA_OUTPUT=/tmp/vidur-agz-spot-worker-user-data.sh
AWS_SPOT_ON_DETAILS/render_ec2_user_data.sh
```

The renderer obtains the coordinator ED25519 host key with `ssh-keyscan`,
creates a `0600` temporary file, and rejects output larger than EC2's 16 KiB
raw user-data limit. Never commit or upload the rendered file outside the EC2
launch template because it contains the private key. The source bootstrap and
renderer contain no credentials.

The launch template must attach `VidurAGZSpotWorkerProfile`; that role only
pulls the immutable worker image from ECR. On boot, the worker verifies SSH to
the coordinator, optionally stages the parent dataset, and starts the
resource-aware worker service.

# XL4 Spot Coordinator And Validation Host

## Instance

- Name: `AlphaZero_SPOT_ON_XL4`
- Purpose: Spot-fleet AlphaGoZero coordinator, ARM64 image validation, and worker development
- Instance ID: `i-0fc135f80a6fa2a3e`
- Public IPv4: `13.221.38.228`
- Private DNS: `ip-172-31-36-224.ec2.internal`
- SSH user: `ubuntu`
- SSH key: `~/.ssh/shaz-pr1.pem`
- Verified hostname: `ip-172-31-36-224`
- Verified OS: Ubuntu 26.04 LTS
- Verified architecture: `aarch64`
- Verified CPU count: `96`
- Verified memory: `185.3 GiB`
- Verified root disk: `2.0 TiB`

## Image

- Image: `vidur-agz-exp3-worker:spot-pull-v3-20260729`
- Deployment status: loaded and independently smoke-tested
- Validation date: `2026-07-29`
- Source commit: `3892281ac70d5685dd0c19b3f2a01d35afe4ea74`
- Runtime overlay SHA-256:
  `06014061e140f3d4fbe0c35b966d8419a0b3b868f49370ef7e8853ec7cd6163b`
- Target platform: `linux/arm64`
- Image ID: `sha256:f46c49853be32ff3b48fc0cf946b9bab14f5a9d6042822c8a039682304818097`
- Docker-reported runtime size: `1,419,630,530 bytes`
- Docker CLI virtual size: `7.67 GB`
- Application path: `/home/ubuntu/vidur-classical-search`
- Build-context artifact: `artifacts/vidur-agz-exp3-worker-build-context-3892281.tar.zst`
- Build-context SHA-256: `6401adb787f2f41a980e43982c77a9d7f7cc51495b883cdae40c39515c168998`

## Resource-Aware Image

- Image: `vidur-agz-exp3-worker:resource-aware-v1-20260731`
- Build date: `2026-07-31`
- Target platform: `linux/arm64`
- Image ID:
  `sha256:6f699a3eb57b697a49f75ed2f0044f05443f716a34ed9a67575ed159c6e9211a`
- Docker-reported size: `1,419,662,281 bytes`
- Base image:
  `vidur-agz-exp3-worker:spot-pull-v3-20260729`
- Scheduler revision: `resource-aware-v1`
- Scheduler/dispatch tests: `14/14` passed inside the ARM64 image
- Native/profile/runtime smoke: passed inside the ARM64 image
- Native code and model inference are unchanged from the validated base image
- 96-core host probe: 90 CPU slots, 182 RAM slots, 90 effective slots
- Simulated 32-core/64-GiB probe: 30 CPU slots, 63 RAM slots, 30 effective slots
- Simulated 16-core/10-GiB probe: 14 CPU slots, 9 RAM slots, 9 effective slots
- Build context: `AWS_SPOT_ON_DETAILS/resource_aware_context`

## Markov Resource-Aware Image

- Image: `vidur-agz-exp3-worker:markov-resource-aware-v2-20260802`
- Build and validation date: `2026-08-02`
- Target platform: `linux/arm64`
- Image ID: `sha256:9b709b29f8735b4a23d127a954d5b185b742ce64daf4276503090eeaf822f604`
- Docker-reported size: `1,419,772,484 bytes`
- Source revision: `3892281ac70d5685dd0c19b3f2a01d35afe4ea74+overlay.dd4099dcf1e1`
- Runtime overlay SHA-256: `dd4099dcf1e152340f20eca70d94d074578e07d4b23ad79f7c413c7405c84897`
- Markov-v2, Spot scheduler, and dispatch tests: `22/22` passed inside the ARM64 image
- Python/native Markov value and both policy parity: exact, maximum absolute error `0.0`
- Runtime native/profile/resource smoke: passed
- 96-core host probe: 90 CPU slots, 180 RAM slots, 90 effective slots
- Build context: `AWS_SPOT_ON_DETAILS/build_context`
- ECR repository: `388660028044.dkr.ecr.us-east-1.amazonaws.com/vidur-agz-exp3-worker`
- ECR tag: `markov-resource-aware-v2-20260802`
- ECR index digest: `sha256:9b709b29f8735b4a23d127a954d5b185b742ce64daf4276503090eeaf822f604`
- ECR ARM64 manifest digest: `sha256:dd8ac94d2aaa256d854287ae48095c3198fb1a34d00a9e29b6fbefa43004162d`
- ECR status: pushed and active; scan-on-push is in progress

## Controller-Configured Fleet Image

- Image tag: `controller-config-v3-20260802`
- Target platform: `linux/arm64`
- Local image ID and ECR digest:
  `sha256:7b47d8ad91e1472949f81bec85ac9154dd286ff731679f7e70b55e4ae37ce2e4`
- Docker-reported runtime size: `1,419,773,449 bytes`
- Protocol schema: `3`; self-play configuration schema: `1`
- Coordinator owns all self-play game/search parameters and fingerprints the
  complete configuration before assignment.
- Coordinator owns complete evaluation commands; workers verify the command
  SHA-256 before execution.
- Focused coordinator/worker tests: `17/17` passed locally and inside ARM64.
- Native/profile/container smoke: passed inside ARM64.
- Launch template: `vidur-agz-spot-worker-canary`
  (`lt-0497e8268243950c9`), production version `4`.

## Controller-Configuration Canary

- Validation date: `2026-08-03` UTC
- Instance: `i-016031b58112ff041`, Spot request `sir-b6mzhv6k`
- Instance type: `c7g.4xlarge`; worker capacity: 14 safe game slots
- Isolated root:
  `/home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/AWS_SPOT_CONTROLLER_CONFIG_CANARY`
- Evaluation command fingerprint:
  `8650703129946bfc19f70fe76df435064077b94eda0756726ac225c727e62ae2`
  (recorded and recomputed values matched).
- Eval command used coordinator values: 13 iterations, discount 0.911,
  3 rollouts, 0.09-second horizon, and PUCT 1.7.
- Self-play configuration fingerprint:
  `73df9a953bd0fae84c283732267668926795f0223344eb4ed9cb82385fe2c0c6`.
- Self-play used coordinator values: 17 iterations, discount 0.913,
  2 rollouts, 0.07-second horizon, PUCT 2.5, alpha 0.071, epsilon 0.19,
  and a two-state shard threshold.
- The launch image deliberately supplied conflicting defaults (999 iterations,
  discount 0.501, full-tree mode, and a 777-state threshold); none reached the
  native self-play command.
- Eval completed on the first attempt. Self-play shards carried the exact
  configuration fingerprint; FIFO replay admitted every controller/adversary
  state with zero drops.
- Final state: worker/coordinator stopped, the canary terminated, and Spot
  request `sir-b6mzhv6k` cancelled.

## Spot Worker IAM

- Role: `arn:aws:iam::388660028044:role/VidurAGZSpotWorkerRole`
- Instance profile: `arn:aws:iam::388660028044:instance-profile/VidurAGZSpotWorkerProfile`
- Trust principal: `ec2.amazonaws.com`
- Attached policy: `arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly`
- Permission simulation: ECR pull operations allowed; `ecr:PutImage` denied
- Status: ready for use by the Spot launch template

## Spot Worker Launch Bootstrap

- Script: `AWS_SPOT_ON_DETAILS/ec2_spot_worker_user_data.sh`
- Image selection: immutable ECR digest `sha256:7b47d8ad91e1472949f81bec85ac9154dd286ff731679f7e70b55e4ae37ce2e4`
- Runtime: Docker container managed by `vidur-agz-spot-worker.service`
- Identity: IMDSv2 EC2 instance ID mapped to a stable worker ID
- Coordinator transport: strict SSH using a launch-template embedded key and pinned `known_hosts`; Secrets Manager remains optional for later hardening
- Persistent state: `/var/lib/vidur-agz/simulator_output`
- Capacity: worker reports live CPU/RAM and XL grants bounded self-play or eval leases
- Validation: outer script syntax, generated runner syntax, positive dry-run, and three fail-closed negative tests passed
- Renderer: `AWS_SPOT_ON_DETAILS/render_ec2_user_data.sh`
- Remaining launch-template input: coordinator host, experiment root, local SSH key path, and optional parent dataset details

## EC2 Spot Canary

- Validation date: `2026-08-02` (`2026-08-03` UTC)
- Instance ID: `i-00a2069e1e28a1e06`
- Spot request ID: `sir-xeafk1bj`
- Instance type and location: `c7g.4xlarge`, `us-east-1d`
- Public IPv4: `35.172.138.23`
- Launch template: `vidur-agz-spot-worker-canary` (`lt-0497e8268243950c9`)
- Default launch-template version: `2` (`ownership-fix-20260802`)
- Isolated coordinator root: `/home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/AWS_SPOT_PULL_CANARY`
- Resource detection: 16 vCPUs, 2 reserved cores, 14 effective game slots, and about 31.8 GiB available RAM
- Scheduler validation: the coordinator limited the worker to one concurrent self-play game despite its 14-slot capacity
- Data-path validation: nine native games completed, nine shards were acknowledged, and all controller/adversary states were admitted by FIFO replay with zero drops
- Bootstrap fix: persistent simulator-output parent directories are created as UID/GID `1000` before the container starts
- Final state: the one-time Spot canary was terminated after validation; its worker service and the isolated coordinator were stopped first

## Validation

- Docker Engine: `29.1.3`
- Docker buildx: `0.30.1`
- Container architecture: `aarch64`
- Python: `3.12.13`
- Native module: imported successfully as
  `mcts_native_gv2.cpython-312-aarch64-linux-gnu.so`
- Vidur execution-cache files: `433`
- Prefill profile SHA-256:
  `47f2ed85aca4a5ec47eacae8425d38284e76759b75f5c069852d393ecc62329b`
- Decode profile SHA-256:
  `b14044faaa5f9f5fea1b159f5bd031ce8538b30b59eb8acd32cd9625093c4d2a`
- Worker and arena command imports: passed
- Discounted-return native smoke check: passed
- Original pull-protocol tests: `9/9` passed
- Isolated eval canary:
  - controller assigned controller `v104` and adversary `v101`
  - native game completed in `6.328475` seconds at 10 MCTS iterations
  - every uploaded file passed SHA-256 verification
  - the production distributed merger accepted the result
- Isolated self-play canary:
  - controller returned self-play after eval finalization
  - exact controller `v104` and adversary `v101` models were used
  - two shards were acknowledged and ingested with `admit_fifo`
  - no ready-shard backlog remained
- The live EXP3 root was not used for canary queue or replay writes
- Host disk after build and load: approximately `1.9 TiB` available

XL4 is the durable coordinator for the production Spot experiment. Replay,
promoted models, parent datasets, and accepted outputs live on its 2 TiB EBS
volume and remain external to ephemeral worker images.

## Atomic Replay Snapshot Fix

On `2026-08-04`, the first production training attempt overlapped replay
partition publication. The state CSV became visible before its policy CSV,
causing indexed sampling to reject the partition as incomplete.

Training now snapshots only partitions containing the atomically published
`partition_manifest.json` commit marker and both replay CSVs. State and policy
paths are captured together, so concurrent ingestion cannot produce a mixed
snapshot. In-progress partitions are deferred to the next training cycle rather
than failing the current cycle.

- Fixed ECR tag: `adaptive-puct05-2thread-v3-racefix-20260804`
- ECR digest: `sha256:9e6470ee66ca5459b659e32e5f775f46fa27dc8658145f5a17ec389b2c8653b3`
- The live XL4 coordinator and host source were hot-patched without restart.

## Diversified Spot Fleet

- Auto Scaling group: `vidur-agz-exp3-spot-workers-v1`
- Fleet specification: `AWS_SPOT_ON_DETAILS/exp3_spot_asg.json`
- Launch template: `lt-0497e8268243950c9`, production version `4`
- Capacity unit: weighted vCPUs; every override weight equals its vCPU count
- Initial/minimum capacity: `0`; no instances launch during configuration
- Maximum configured capacity: `3000` weighted vCPUs
- Purchase option: 100% Spot, with `price-capacity-optimized` allocation and
  Capacity Rebalancing enabled
- Instance pools: ARM64 `c7g`, `c8g`, and `c7gn`, from 16 to 96 vCPUs
- Availability Zones: `us-east-1a`, `1b`, `1c`, `1d`, and `1f`
- Standard Spot vCPU quota before this fleet: `640`; applied quota after the
  approved increase: `3000`
- Production coordinator: `AlphaZero_SPOT_ON_XL4` (`13.221.38.228`)
- Production experiment root:
  `/home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/AlphaGoZero_SPOT_UNTRAINED_.98_500_MCTS_3s_10ROLLOUT_DNN_2400VCPU`
- A target of `2400` weighted units requests 2,400 provisioned vCPUs. Because
  each worker reserves two cores, approximately 2,400 safe game slots require
  about `2752` provisioned vCPUs instead.

## Production 2,400-vCPU Test

Verified snapshot: `2026-08-03 05:15 UTC`.

- Experiment root:
  `/home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/AlphaGoZero_SPOT_UNTRAINED_.98_500_MCTS_3s_10ROLLOUT_DNN_2400VCPU`
- Coordinator container: `vidur-agz-spot-coordinator` on XL4
- Starting models: untrained DNN controller/adversary version `100`
- Self-play: 500 MCTS iterations, 10 policy rollouts per leaf, 3-second
  rollout horizon, 5-second game/sample window, and discount `0.98`
- Self-play shard threshold: 1,000 states
- Search configuration SHA-256:
  `f6c8ad218a035b32c81c00f7ca745c2ce2713df69ca615914eb465860a56a5a8`
- Evaluation: 500 iterations, 5 rollouts, 140 role games, and the configured
  81-win promotion threshold
- Fleet state: desired/fulfilled `2400` weighted vCPUs, 41/41 instances
  in service, Capacity Rebalancing enabled
- Effective scheduler capacity: 2,249 safe game slots; 2,249 assigned and
  151 unavailable because each worker reserves OS/uploader capacity
- Active instance mix: 18 `c7g.16xlarge`, 7 `c7g.12xlarge`, 2
  `c7g.8xlarge`, 3 `c8g.24xlarge`, 6 `c8g.16xlarge`, 2 `c8g.12xlarge`,
  2 `c8g.8xlarge`, and 1 `c8g.4xlarge`
- Availability Zones in use: `us-east-1a`, `1b`, `1c`, `1d`, and `1f`
- XL4 disk snapshot: 153 GiB used, approximately 1.8 TiB free
- Clean-start verification: replay and incoming queues were empty before the
  production fleet started; workers received only the controller-owned config

The preceding isolated quick canary is archived at:

```text
/home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/
AlphaGoZero_SPOT_UNTRAINED_.98_500_MCTS_3s_10ROLLOUT_DNN_2400VCPU_CANARY_VALIDATION_20260803_0443Z
```

It produced 20/20 accepted shards and 1,504 states with no checksum,
configuration-fingerprint, upload, or replay-admission rejection.

### Failure Recovery

- Worker heartbeat interval: 60 seconds.
- Coordinator heartbeat/lease cap: 300 seconds for both self-play and eval.
- A dead worker's scheduler reservation expires no later than five minutes
  after its final accepted heartbeat; the next poll reassigns that capacity.
- Eval retries use a new lease token. Results from an expired token are
  rejected, preventing a late interrupted worker from overwriting a retry.
- Self-play is generated only from XL's current promoted model manifest.
  Uploaded shards are checksum-verified and acknowledged before workers delete
  them. Already accepted data is unaffected by an instance interruption.
- The only expected loss is work that existed solely on the interrupted
  instance: running games and an incomplete shard below 1,000 states.
- ASG Capacity Rebalancing may launch a replacement before AWS terminates a
  rebalance-risk instance. EC2 boot/image-pull time is outside the five-minute
  coordinator lease guarantee.

The five-minute cap was enabled live at `2026-08-03 05:21:26 UTC` without
restarting the coordinator or any self-play game. Reconfiguration immediately
shortened all 42 leases present during a Capacity Rebalancing overlap. Worker
`spot-i-0303ce4b30f2f43b4` did not renew; its stale reservation expired and was
reclaimed by `spot-i-08e0b64f2b8241195` at `05:26:42 UTC`. The ASG then
returned to 41 in-service workers and the scheduler returned to 2,249 active
safe slots. The active search fingerprint remained unchanged.

Validation results for this change:

- Local scheduler and worker-dispatch tests: `21/21` passed.
- XL4 in-container scheduler protocol tests: `13/13` passed.
- Bootstrap script syntax and Git whitespace checks: passed.
- Every live lease carried `worker_heartbeat_timeout_sec=300`; observed lease
  time remaining never exceeded 300 seconds after activation.
- XL4 pre-change code backup:
  `AWS_SPOT_ON_DETAILS/backups/pre_5min_lease_20260803T0515Z`
- Pre-change scheduler configuration backup:
  `spot_work/control/scheduler_config.pre_5min_20260803T0515Z.json`

### Cost Snapshot

The verified live instance mix was estimated at approximately `$26.53/hour`
for Spot worker compute and `$0.45/hour` for worker EBS. XL4 adds about
`$3.83/hour` compute and `$0.22/hour` EBS, for approximately `$31.03/hour`
all-in. The same worker mix at On-Demand rates plus XL4 was approximately
`$94.58/hour`, a savings of about `$63.55/hour` or `67%` overall. Spot prices
vary by AZ and time, so this is a launch-time estimate rather than a billing
guarantee.

### Operations

Inspect ASG capacity:

```bash
aws --profile vidur-spot --region us-east-1 autoscaling \
  describe-auto-scaling-groups \
  --auto-scaling-group-names vidur-agz-exp3-spot-workers-v1
```

Inspect scheduler capacity and reclaim expired leases:

```bash
docker exec vidur-agz-spot-coordinator bash -lc \
  'cd /home/ubuntu/vidur-classical-search && \
   .venv/bin/python3 -m vidur.AlphaGoZero.spot_work_cli \
   --root /home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/AlphaGoZero_SPOT_UNTRAINED_.98_500_MCTS_3s_10ROLLOUT_DNN_2400VCPU \
   scheduler-status'
```

Scale down safely by setting ASG desired capacity to zero. The coordinator and
its EBS data remain running unless explicitly stopped:

```bash
aws --profile vidur-spot --region us-east-1 autoscaling \
  update-auto-scaling-group \
  --auto-scaling-group-name vidur-agz-exp3-spot-workers-v1 \
  --desired-capacity 0
```

## Dynamic-Noise 2K Spot Experiment

Launch snapshot: `2026-08-06 21:01 UTC`.

- Experiment root:
  `/home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/AlphaGoZero_SPOT_UNTRAINED_.98_2k_MCTS_3s_p95_DEPENDENCE_LEAF_RELATIVE_1ROLLOUT_DNN_MARKOV_1THREAD_Cpuct0p5_DIRICHLET12_OVER_N_SAMPLE15_2400VCPU`
- Starting models: fresh untrained DNN controller/adversary version `100`.
- Search: 2,000 MCTS iterations, one 3-second rollout per leaf, one thread per
  game, discount `0.98`, and PUCT `0.5` in self-play, arena, and SJF.
- Exploration: epsilon `0.25`, per-state Dirichlet alpha
  `12 / canonical_action_count`, and visit sampling for the first 15 actions.
- Shard threshold: 1,000 states per worker upload.
- Worker image digest:
  `sha256:ad26d2c8e2f109ad9ed03f4f3ec279365308dc96b6413860d998f5bd347ecbdf`.
- Launch template: `lt-0497e8268243950c9`, replacement version `13`.
- The initial live fleet booted from version `12`. Version `13` adds a shared,
  prewarmed Matplotlib font cache for future replacement workers and was
  activated without recycling or interrupting the live fleet.
- Fleet: 2,400 weighted vCPUs across 42 workers; 2,246 safe one-thread game
  slots registered and assigned at startup.
- Scheduler configuration SHA-256:
  `7270dae8da5d1354ed8c8ec7fe6346abfecfa41cca8a2b9a488e58b87e1e8efc`.

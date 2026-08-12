# AlphaGoZero Spot Pipeline Process Map

This map records configured concurrency for the Spot coordinator and fleet.
Counts are process/thread ceilings; available CPU and RAM can reduce effective
concurrency.

## Coordinator

| Pipeline stage | Concurrency | Implementation |
|---|---:|---|
| Coordinator ingest/control loop | 1 process | xl_coordinator.py |
| Replay index building | Up to 80 processes | AGZ_REPLAY_INDEX_BUILD_WORKERS=80 |
| Replay extraction/materialization | Up to 80 processes per sequential extraction group | AGZ_REPLAY_EXTRACTION_WORKERS=80 |
| Policy cache building | Up to 80 processes | AGZ_POLICY_CACHE_BUILD_WORKERS=80 |
| Controller/adversary value fitting | 2 concurrent model jobs, 40 Torch threads each | AGZ_DNN_TORCH_THREADS_PER_MODEL=40 |
| Controller/adversary policy fitting | 2 concurrent model jobs, 40 Torch threads each | AGZ_DNN_TORCH_THREADS_PER_MODEL=40 |
| Post-fit policy metrics | 80 total processes shared across both roles | AGZ_POLICY_METRICS_WORKERS=80 |
| Evaluation orchestration | 1 coordinator process; games execute on workers | Spot pull queue |

Post-fit policy metrics previously performed two serial full-dataset prediction
passes and took about 14m 30s. The multiprocess implementation partitions both
roles into balanced root ranges, performs inference in one shared 80-process
pool, and reduces sums/counts into the original MSE, cross-entropy, top-1, and
top-3 definitions. The target wall time is at most 2 minutes on the 96-vCPU
coordinator; confirm this target from the first deployed candidate's
train_candidate_policy_metrics_complete progress event.

The 80-process limit is global for this stage. It does not create 80 controller
plus 80 adversary processes.

## Spot Workers

| Work type | Per-game threads | Fleet limit | Notes |
|---|---:|---:|---|
| Self-play | 2 | 1,200 games | Further capped by each worker's CPU and RAM |
| Arena evaluation | 4 | 1,200 games | Evaluation queue has priority |
| SJF evaluation | 4 | 1,200 games | Created only when the pipeline requests SJF |

Worker capacity is:

min(floor((available_vCPUs - reserved_vCPUs) / threads_per_game), available_RAM_GiB)

The worker reserves max(2, ceil(vCPUs / 16)) CPU cores unless explicitly
overridden.

## Evaluation Preemption

An evaluation batch now preempts active self-play instead of waiting for a
complete self-play wave:

1. begin_eval_batch writes the active evaluation record and queues games.
2. Each self-play heartbeat checks the active evaluation record.
3. The coordinator returns an explicit preempt_requested response while keeping
   the lease valid.
4. The worker's cancellation event stops new launches, collects games that
   already exited successfully, freezes/uploads completed replay rows, and
   terminates only unfinished games.
5. The worker releases its self-play lease with status preempted.
6. Its next pull receives arena work. Self-play resumes automatically after the
   evaluation batch finishes.

Spot workers use a 10-second heartbeat, so normal handoff latency is bounded by
roughly 10 seconds plus process termination/upload time. Network or heartbeat
failures do not trigger cancellation; only an authenticated coordinator
preempt_requested response does.

## Transport Failure Protection

Worker/controller SSH calls retry up to 12 times with exponential backoff from
1 to 30 seconds. Result publication retries up to 8 times. A completed game is
kept on the worker until the coordinator acknowledges it; publication failure
does not consume a game retry or terminate the worker daemon.

Result publication is idempotent. If SSH disconnects after the atomic remote
move, the next attempt verifies and accepts the existing result. Before leases
are expired or reassigned, the coordinator reconciles checksum-valid published
results against their task manifest. Only tasks without valid output are
requeued. Worker errors and backoff events are recorded in
`spot_worker_events.jsonl`.

## Deployment State

These settings are present in local source and Spot image/build templates. They
do not affect an already running fleet until a new image and launch-template
version are deployed. Do not update the active ASG during an ongoing experiment
without an explicit deployment window.

## Dynamic-Noise 2K Experiment

The fresh Spot experiment uses one CPU thread per game, 2,000 MCTS
simulations, one rollout per leaf, PUCT `0.5`, epsilon `0.25`, and dynamic
Dirichlet alpha `12 / canonical_action_count`. The fleet-wide scheduler caps
self-play and evaluation at 2,400 concurrent games; per-worker CPU reservation
and RAM checks reduce the effective count when required. Workers publish every
1,000 states. Visit-temperature sampling is limited to the first 15 actions.

The active launch uses one thread for self-play, arena evaluation, and SJF
evaluation. At 2,400 provisioned vCPUs, the initial 42-worker fleet registered
2,246 resource-safe one-thread game slots after reserving 154 cores for worker
operating-system and upload duties.

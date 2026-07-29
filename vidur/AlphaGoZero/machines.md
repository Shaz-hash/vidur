# AlphaGoZero Machine Inventory

This file records the machines assigned to each active or planned AlphaGoZero
experiment. Do not reuse a machine for another experiment unless its existing
coordinator, worker daemon, and evaluation jobs have been stopped first.

## Active: `.995` / 1K MCTS / Simple Bigger Replay

Last verified: `2026-07-12 07:00 UTC`

| Field | Value |
| --- | --- |
| Run ID | `agz_995_1k_simple35m_v100` |
| Output root | `/home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/AlphaGoZero_.995_1k_MCTS_simple_bigger_replay` |
| Discount factor | `0.995` (current Python and native default) |
| MCTS simulations | `1,000` for self-play and evaluation |
| Root exploration | Dirichlet `alpha=0.1`, `epsilon=0.35`; PUCT `2.5` |
| Replay capacity | Plain FIFO: controller `20M`, adversary `15M`, total `35M` |
| Training gate | At least `600,000` newly accepted states |
| Policy samples | Controller `250k`; adversary `250k` |
| Current promoted versions | Controller `v100`; adversary `v100` |
| Worker sampling | 60 parallel games per worker, 10,000-state upload buffer, first 30 moves sampled |

The coordinator trains and evaluates candidate controller and adversary policy/value
models separately. Workers continuously generate and upload self-play shards; XL
eagerly drains the durable transfer handoff into the role FIFO replay.

| Role | SSH alias | Active process | Assignment |
| --- | --- | --- | --- |
| XL coordinator | `bellman-classical-xl` | `xl_coordinator --loop --poll-sec 20` | Ingests shards, maintains replay, trains models, runs and coordinates evaluation. |
| Worker 1 | `bellman-classical-worker-1` | `worker_daemon` | Self-play; parent states `server_01`, hops `151-300`; game IDs start at `410000000`. |
| Worker 2 | `bellman-classical-worker-2` | `worker_daemon` | Self-play; parent states `server_02`, hops `301-450`; game IDs start at `420000000`. |
| Worker 3 | `bellman-classical-worker-3` | `worker_daemon` | Self-play; parent states `server_03`, hops `451-600`; game IDs start at `430000000`. |
| Worker 4 | `bellman-classical-worker-4` | `worker_daemon` | Self-play; parent states `server_04`, hops `601-750`; game IDs start at `440000000`. |
| Worker 5 | `bellman-classical-worker-5` | `worker_daemon` | Self-play; adversary-root parent states `server_00`, hops `0-100`; game IDs start at `450000000`. |
| Worker 6 | `bellman-classical-worker-6` | `worker_daemon` | Self-play; adversary-root parent states `server_01`, hops `100-200`; game IDs start at `460000000`. |
| Worker 7 | `bellman-classical-worker-7` | `worker_daemon` | Self-play; adversary-root parent states `server_02`, hops `200-300`; game IDs start at `470000000`. |
| Worker 8 | `bellman-classical-worker-8` | `worker_daemon` | Self-play; adversary-root parent states `server_03`, hops `300-400`; game IDs start at `480000000`. |

## Planned Experiments

Add each new cluster below before deployment. Record its output root, discount,
MCTS iteration count, coordinator alias, worker aliases, and whether any machine is
shared with another active experiment.

### EXP2 `.995` / 2K Self-Play + 1K Eval / Incremental DNN / FIFO Replay

Status: active; launched `2026-07-12 08:53 UTC`. This cluster is independent of
`bellman-classical-xl` and its workers; no EXP2 machine is shared with that
experiment.

| Field | Value |
| --- | --- |
| Purpose | Incremental AlphaGoZero training of separate controller/adversary value and policy residual DNNs. |
| Discount factor | `0.995` for Python MCTS, native MCTS, replay targets, training, and evaluation. |
| MCTS simulations | `2,000` for self-play; `1,000` for arena and SJF evaluation. |
| Self-play exploration | PUCT `2.5`; Dirichlet `alpha=0.05`, `epsilon=0.25`; visit-temperature sampling for the first `20` actions, then deterministic most-visits. |
| Initial models | DNN `v100` distilled from classical `v100`; both value copies start bit-identical and all four artifacts include AdamW optimizer state. |
| Model updates | Every candidate continues from the latest published candidate checkpoint and optimizer state; failed checkpoints accumulate, while self-play remains on the promoted role bundle. |
| Replay | Plain role FIFO: controller `20M`, adversary `15M`; no version quotas. |
| Training gate | `600k` newly accepted states; value `700k`, controller policy `250k`, adversary policy `250k`. |
| Runtime | Arm64 CPython `3.12.13`, SIMD native residual-MLP inference built locally on every host. |
| Calibration | Shared validated A100/a100_dgx profile: 32 rows (`128-4096`), 4,096-token time `0.3866926276s`; identical cache and profile hash on every host. |
| Coordinator | `bellman-classical-exp2-xl` (`13.216.249.51`, `i-087df4a3785ad3359`) |
| Workers | Eight dedicated `bellman-classical-exp2-worker-*` hosts listed below. |
| Output root | `/home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/AlphaGoZero_.995_1k_MCTS_simple_bigger_replay_DNN` |

| Role | SSH alias | Public IP | EC2 instance ID | Designated work |
| --- | --- | --- | --- | --- |
| XL coordinator | `bellman-classical-exp2-xl` | `13.216.249.51` | `i-087df4a3785ad3359` | Replay ingestion, training, model promotion, and evaluation coordination. |
| Worker 1 | `bellman-classical-exp2-worker-1` | `3.90.37.53` | `i-0bd10b5b8b03ea92e` | Self-play sample generation and distributed evaluation. |
| Worker 2 | `bellman-classical-exp2-worker-2` | `98.85.254.192` | `i-0d62e6f5e9b1b1fce` | Self-play sample generation and distributed evaluation. |
| Worker 3 | `bellman-classical-exp2-worker-3` | `18.234.151.97` | `i-03bb6087f43774c78` | Self-play sample generation and distributed evaluation. |
| Worker 4 | `bellman-classical-exp2-worker-4` | `54.80.157.141` | `i-0a894967fcc876c0a` | Self-play sample generation and distributed evaluation. |
| Worker 5 | `bellman-classical-exp2-worker-5` | `3.81.66.112` | `i-02258001cde058e0b` | Self-play sample generation and distributed evaluation. |
| Worker 6 | `bellman-classical-exp2-worker-6` | `3.81.93.125` | `i-09c1f39d4b73b4fa3` | Self-play sample generation and distributed evaluation. |
| Worker 7 | `bellman-classical-exp2-worker-7` | `3.94.254.84` | `i-0abdc0c0a3392c779` | Self-play sample generation and distributed evaluation. |
| Worker 8 | `bellman-classical-exp2-worker-8` | `52.207.171.193` | `i-03b75a9eb5fcc2021` | Self-play sample generation and distributed evaluation. |

## EXP3: Leaf-Relative 1-Second Policy Rollout / 500 MCTS / Incremental DNN

Status: relaunch prepared `2026-07-22` after replacing the original rollout
deadline with the expansion-leaf-relative protocol.
This is an independent cluster and shares no machines or output directories with
EXP2.

| Field | Value |
| --- | --- |
| Purpose | Test expansion-leaf-relative, policy-guided 1-second rollouts in native AlphaGoZero while retaining Python/native parity. |
| Output root | `/home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/AlphaGoZero_EXP3_UNTRAINED_.995_500_MCTS_1s_LEAF_RELATIVE_ROLLOUT_DNN` |
| Model | Separate incremental controller/adversary residual-DNN policy and value models, Markov-v2 value features. |
| Discount | `0.995` throughout self-play, replay targets, training, and evaluation. |
| Search | `500` MCTS iterations; `10` policy-guided continuations per expanded leaf. Sibling actions share `deadline = expanded_leaf_time + 1.0s`; the selected action consumes part of that horizon and continuation stops at/crosses the deadline. |
| Exploration | PUCT `2.5`; Dirichlet `alpha=0.05`, `epsilon=0.25`; first `20` moves sampled. |
| Training gate | `50k` newly accepted states, at least `100k` controller and `50k` adversary states. |
| Policy samples | Controller `100k`; adversary `50k`. |
| Replay | Plain role FIFO: controller `20M`, adversary `15M`. |
| Worker load | 60 concurrent self-play processes per worker; one rollout thread per process; shards publish every 4,000 states. |
| Runtime reference | EXP2 matching-role host, CPython `3.12.13`, Arm64 native build, A100 profile SHA-256 `47f2ed85...329b`. |
| 60-process benchmark | One thread: median `6.57s`, p95 `7.20s`, `4.11` searches/s, `140.1 GiB` peak RSS. Two threads: median `4.69s` but the same `4.10` searches/s. Four threads regressed to `2.53` searches/s. One thread was selected for aggregate throughput and headroom. |
| Evaluation layout | Self-play pauses first. XL uses cores `4-95` and workers use `2-95`; role games use two rollout threads with caps of `46` and `47` games respectively. SJF uses eight threads per game. |

| Role | SSH alias | Public IP | EC2 instance ID | Designated work |
| --- | --- | --- | --- | --- |
| XL coordinator | `bellman-classical-exp3-xl` | `3.92.29.240` | `i-02f4e2ed0c772c534` | Replay ingestion, DNN training, distributed evaluation, and promotion. |
| Worker 1 | `bellman-classical-exp3-worker-1` | `54.146.143.25` | `i-0e90f4613c2b78ad7` | Self-play and distributed evaluation. |
| Worker 2 | `bellman-classical-exp3-worker-2` | `54.166.214.226` | `i-01b2eec16c274294e` | Self-play and distributed evaluation. |
| Worker 3 | `bellman-classical-exp3-worker-3` | `3.82.3.46` | `i-014435c9a4b58835c` | Self-play and distributed evaluation. |
| Worker 4 | `bellman-classical-exp3-worker-4` | `54.162.204.223` | `i-0d753bcfb659402f3` | Self-play and distributed evaluation. |
| Worker 5 | `bellman-classical-exp3-worker-5` | `54.81.147.54` | `i-0c40daae4c3066233` | Self-play and distributed evaluation. |
| Worker 6 | `bellman-classical-exp3-worker-6` | `54.163.82.13` | `i-01ffb145e88eba394` | Self-play and distributed evaluation. |
| Worker 7 | `bellman-classical-exp3-worker-7` | `52.91.155.208` | `i-05bf2fbcc92c0ed4c` | Self-play and distributed evaluation. |
| Worker 8 | `bellman-classical-exp3-worker-8` | `54.81.120.20` | `i-00b262b9fc57d1635` | Self-play and distributed evaluation. |

Provisioning is inventory-driven by
`settingUpServers/agz_cluster_setup.py` and
`settingUpServers/agz_clusters.json`. The script copies the validated runtime,
execution cache, profile, and matching parent-state dataset from EXP2, but always
syncs current source from the local repository and rebuilds the native extension.

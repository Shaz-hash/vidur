from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence

from .compat_export import CompatExportPaths, export_compat_csvs
from .config import SamplerRunConfig, round_dir
from .sampler_worker import SamplerWorkerResult, SamplerWorkerTask, run_sampler_worker
from .storage import MergeOutput, WorkerShardPaths, merge_worker_shards


@dataclass(frozen=True)
class SamplerRoundResult:
    round_idx: int
    worker_results: Sequence[SamplerWorkerResult]
    merge_output: MergeOutput
    compat_output: CompatExportPaths


def _build_tasks(cfg: SamplerRunConfig, round_idx: int) -> List[SamplerWorkerTask]:
    hops = list(cfg.collection.history_hops) or [0]
    tasks: List[SamplerWorkerTask] = []
    for wid in range(int(cfg.collection.workers)):
        hop = int(hops[wid % len(hops)])
        tasks.append(
            SamplerWorkerTask(
                worker_id=int(wid),
                round_idx=int(round_idx),
                seed=int(cfg.seed + round_idx * 100_000 + wid),
                history_hop=hop,
                out_dir=str(cfg.output.out_dir),
                cfg=cfg,
            )
        )
    return tasks


def _build_tasks_with_target(
    cfg: SamplerRunConfig,
    *,
    round_idx: int,
    per_worker_target: int,
) -> List[SamplerWorkerTask]:
    tasks = _build_tasks(cfg, round_idx)
    if per_worker_target <= 0:
        return tasks
    out: List[SamplerWorkerTask] = []
    for task in tasks:
        out.append(
            SamplerWorkerTask(
                worker_id=task.worker_id,
                round_idx=task.round_idx,
                seed=task.seed,
                history_hop=task.history_hop,
                out_dir=task.out_dir,
                cfg=task.cfg,
                target_anchor_samples=int(per_worker_target),
            )
        )
    return out


def _collect_round(
    cfg: SamplerRunConfig,
    round_idx: int,
    *,
    per_worker_target: int = 0,
) -> Sequence[SamplerWorkerResult]:
    tasks = _build_tasks_with_target(cfg, round_idx=round_idx, per_worker_target=int(per_worker_target))
    if len(tasks) <= 1:
        class _LocalQueue:
            def __init__(self) -> None:
                self._item = None

            def put(self, item: Any) -> None:
                self._item = item

            def get(self, timeout: float | None = None) -> Any:
                _ = timeout
                return self._item

        q = _LocalQueue()
        run_sampler_worker(tasks[0], q)
        msg = q.get()
        assert isinstance(msg, SamplerWorkerResult)
        return [msg]

    ctx = mp.get_context("spawn")
    q: Any = ctx.Queue()
    procs = []
    for task in tasks:
        p = ctx.Process(target=run_sampler_worker, args=(task, q), daemon=False)
        p.start()
        procs.append(p)

    results: List[SamplerWorkerResult] = []
    for _ in tasks:
        msg = q.get(timeout=60 * 60)
        if not isinstance(msg, SamplerWorkerResult):
            raise RuntimeError(f"unexpected worker payload type: {type(msg)}")
        results.append(msg)

    for p in procs:
        p.join(timeout=60.0)
        if p.exitcode not in (0, None):
            raise RuntimeError(f"sampler worker crashed: pid={p.pid} exit={p.exitcode}")

    return sorted(results, key=lambda r: r.worker_id)


def run_parallel_sampler(
    cfg: SamplerRunConfig,
    *,
    export_compat_root: bool = False,
    export_compat_iter: bool = False,
) -> Sequence[SamplerRoundResult]:
    out: List[SamplerRoundResult] = []
    target_unique = int(cfg.collection.target_unique_states)
    max_rounds = int(cfg.collection.max_rounds)
    global_anchor_state_ids: set[str] = set()

    for round_idx in range(max_rounds):
        remaining = max(0, int(target_unique - len(global_anchor_state_ids)))
        if remaining <= 0:
            break
        per_worker_target = max(1, (remaining + max(1, int(cfg.collection.workers)) - 1) // max(1, int(cfg.collection.workers)))
        configured_shard_target = int(cfg.collection.shard_unique_states_per_worker)
        if configured_shard_target > 0:
            per_worker_target = min(per_worker_target, configured_shard_target)

        rd = round_dir(cfg, round_idx)
        rd.mkdir(parents=True, exist_ok=True)

        worker_results = list(_collect_round(cfg, round_idx, per_worker_target=per_worker_target))
        failures = [r for r in worker_results if not r.ok]
        if failures:
            err = failures[0]
            raise RuntimeError(
                f"sampler worker {err.worker_id} failed at round={round_idx}:\n{err.error}"
            )

        shards: List[WorkerShardPaths] = []
        for r in worker_results:
            if r.shard is None:
                raise RuntimeError(f"worker {r.worker_id} returned no shard")
            shards.append(r.shard)

        merge_output = merge_worker_shards(
            out_dir=Path(cfg.output.out_dir),
            round_idx=round_idx,
            shards=shards,
            compression=str(cfg.output.parquet_compression),
        )
        for wr in worker_results:
            global_anchor_state_ids.update(str(sid) for sid in wr.accepted_anchor_state_ids if sid)
        global_unique = int(len(global_anchor_state_ids))

        compat_output = CompatExportPaths(mcts_root_compat_csv="", mcts_iter_compat_csv="")
        if export_compat_root or export_compat_iter:
            compat_output = export_compat_csvs(
                merged_dir=Path(cfg.output.out_dir) / f"round_{round_idx:03d}" / "merged",
                out_dir=Path(cfg.output.out_dir) / f"round_{round_idx:03d}" / "compat",
            )
            if not export_compat_root and compat_output.mcts_root_compat_csv:
                Path(compat_output.mcts_root_compat_csv).unlink(missing_ok=True)
                compat_output = CompatExportPaths(
                    mcts_root_compat_csv="",
                    mcts_iter_compat_csv=compat_output.mcts_iter_compat_csv,
                )
            if not export_compat_iter and compat_output.mcts_iter_compat_csv:
                Path(compat_output.mcts_iter_compat_csv).unlink(missing_ok=True)
                compat_output = CompatExportPaths(
                    mcts_root_compat_csv=compat_output.mcts_root_compat_csv,
                    mcts_iter_compat_csv="",
                )

        out.append(
            SamplerRoundResult(
                round_idx=round_idx,
                worker_results=tuple(worker_results),
                merge_output=merge_output,
                compat_output=compat_output,
            )
        )

        print(
            f"[linear.sampler] round={round_idx} merged_unique_states={merge_output.unique_states} "
            f"merged_unique_anchor_samples={merge_output.unique_anchor_samples} "
            f"controller_lp_samples={merge_output.controller_lp_samples} "
            f"adversary_lp_samples={merge_output.adversary_lp_samples} "
            f"global_unique_anchor_samples={global_unique}/{target_unique}",
            flush=True,
        )
        if global_unique >= target_unique:
            break

    return tuple(out)

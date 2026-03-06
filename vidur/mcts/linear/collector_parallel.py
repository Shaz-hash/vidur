from __future__ import annotations

import csv
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np

from .collector_worker import run_collection_worker
from .config import LinearPipelineConfig, WorkerResult, WorkerTask, round_dir


@dataclass(frozen=True)
class CollectedDataset:
    train_features: np.ndarray
    train_targets: np.ndarray
    eval_features: np.ndarray
    eval_targets: np.ndarray
    train_pred_before: np.ndarray
    eval_pred_before: np.ndarray
    train_npz: str
    eval_npz: str
    train_meta_csv: str
    eval_meta_csv: str
    worker_results: Sequence[WorkerResult]


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as d:
        x = np.asarray(d["features"], dtype=np.float32)
        y = np.asarray(d["targets"], dtype=np.float32)
        p = np.asarray(d["pred_before"], dtype=np.float32)
    return x, y, p


def _concat_or_empty(rows: List[np.ndarray], width: int | None = None) -> np.ndarray:
    if not rows:
        if width is None:
            return np.zeros((0,), dtype=np.float32)
        return np.zeros((0, int(width)), dtype=np.float32)
    return np.concatenate(rows, axis=0)


def _merge_meta_csv(inputs: Sequence[Path], out_path: Path) -> None:
    fieldnames = None
    rows: list[dict[str, Any]] = []
    for p in inputs:
        with p.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = list(r.fieldnames or [])
            for row in r:
                rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames or [])
        if fieldnames:
            w.writeheader()
            for row in rows:
                w.writerow(row)


def collect_samples_parallel(
    *,
    cfg: LinearPipelineConfig,
    round_idx: int,
    model_ckpt_path: Path,
) -> CollectedDataset:
    rd = round_dir(cfg, round_idx)
    rd.mkdir(parents=True, exist_ok=True)

    n_workers = int(cfg.collection.workers)
    hops = list(cfg.collection.history_hops) or [0]

    results: list[WorkerResult] = []
    if n_workers <= 1:
        class _LocalQueue:
            def __init__(self) -> None:
                self._item = None

            def put(self, item: Any) -> None:
                self._item = item

            def get(self, timeout: float | None = None) -> Any:
                _ = timeout
                return self._item

        q = _LocalQueue()
        task = WorkerTask(
            worker_id=0,
            round_idx=int(round_idx),
            seed=int(cfg.seed + round_idx * 10000),
            history_hop=int(hops[0]),
            out_dir=str(rd),
            model_ckpt_path=str(model_ckpt_path),
            cfg=cfg,
        )
        run_collection_worker(task, q)
        msg = q.get()
        if not isinstance(msg, WorkerResult):
            raise RuntimeError(f"worker result payload has unexpected type: {type(msg)}")
        results = [msg]
    else:
        ctx = mp.get_context("spawn")
        result_q: Any = ctx.Queue()

        procs = []
        for wid in range(n_workers):
            hop = int(hops[wid % len(hops)])
            task = WorkerTask(
                worker_id=int(wid),
                round_idx=int(round_idx),
                seed=int(cfg.seed + round_idx * 10000 + wid),
                history_hop=hop,
                out_dir=str(rd),
                model_ckpt_path=str(model_ckpt_path),
                cfg=cfg,
            )
            p = ctx.Process(target=run_collection_worker, args=(task, result_q), daemon=False)
            p.start()
            procs.append(p)

        for _ in range(n_workers):
            msg = result_q.get(timeout=60 * 60)
            if not isinstance(msg, WorkerResult):
                raise RuntimeError(f"worker result payload has unexpected type: {type(msg)}")
            results.append(msg)

        for p in procs:
            p.join(timeout=30.0)
            if p.exitcode not in (0, None):
                raise RuntimeError(f"collector worker died: pid={p.pid} exitcode={p.exitcode}")

    failures = [r for r in results if not r.ok]
    if failures:
        first = failures[0]
        raise RuntimeError(
            f"worker {first.worker_id} failed during collection:\n{first.error}"
        )

    train_xs: List[np.ndarray] = []
    train_ys: List[np.ndarray] = []
    train_ps: List[np.ndarray] = []
    eval_xs: List[np.ndarray] = []
    eval_ys: List[np.ndarray] = []
    eval_ps: List[np.ndarray] = []

    train_meta_paths: List[Path] = []
    eval_meta_paths: List[Path] = []

    for r in sorted(results, key=lambda x: x.worker_id):
        tx, ty, tp = _load_npz(Path(r.train_npz))
        ex, ey, ep = _load_npz(Path(r.eval_npz))
        train_xs.append(tx)
        train_ys.append(ty)
        train_ps.append(tp)
        eval_xs.append(ex)
        eval_ys.append(ey)
        eval_ps.append(ep)
        train_meta_paths.append(Path(r.train_meta_csv))
        eval_meta_paths.append(Path(r.eval_meta_csv))

    train_features = _concat_or_empty(train_xs, width=30).astype(np.float32)
    train_targets = _concat_or_empty(train_ys).astype(np.float32)
    train_pred_before = _concat_or_empty(train_ps).astype(np.float32)

    eval_features = _concat_or_empty(eval_xs, width=30).astype(np.float32)
    eval_targets = _concat_or_empty(eval_ys).astype(np.float32)
    eval_pred_before = _concat_or_empty(eval_ps).astype(np.float32)

    train_npz = rd / f"train_samples_round_{round_idx:03d}.npz"
    eval_npz = rd / f"eval_samples_round_{round_idx:03d}.npz"
    train_meta_csv = rd / f"train_samples_round_{round_idx:03d}_meta.csv"
    eval_meta_csv = rd / f"eval_samples_round_{round_idx:03d}_meta.csv"

    np.savez_compressed(
        train_npz,
        features=train_features,
        targets=train_targets,
        pred_before=train_pred_before,
    )
    np.savez_compressed(
        eval_npz,
        features=eval_features,
        targets=eval_targets,
        pred_before=eval_pred_before,
    )

    _merge_meta_csv(train_meta_paths, train_meta_csv)
    _merge_meta_csv(eval_meta_paths, eval_meta_csv)

    return CollectedDataset(
        train_features=train_features,
        train_targets=train_targets,
        eval_features=eval_features,
        eval_targets=eval_targets,
        train_pred_before=train_pred_before,
        eval_pred_before=eval_pred_before,
        train_npz=str(train_npz),
        eval_npz=str(eval_npz),
        train_meta_csv=str(train_meta_csv),
        eval_meta_csv=str(eval_meta_csv),
        worker_results=tuple(sorted(results, key=lambda x: x.worker_id)),
    )

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from vidur.mcts.linear.sampler.storage import read_parquet_records


StateId = str
WorkerId = str
EdgeSig = Tuple[str, str, int, str]  # (state_id, actor, canonical_index, next_state_id)


@dataclass(frozen=True)
class WorkerOverlapData:
    worker: WorkerId
    state_ids: Set[StateId]
    edge_sigs: Set[EdgeSig]
    sample_group_sigs: Set[str]


def _worker_dirs(round_dir: Path) -> List[Path]:
    shards_dir = round_dir / "shards"
    if not shards_dir.exists():
        raise FileNotFoundError(f"Missing shards dir: {shards_dir}")
    out = [p for p in shards_dir.iterdir() if p.is_dir() and p.name.startswith("worker_")]
    if not out:
        raise FileNotFoundError(f"No worker shard dirs found under: {shards_dir}")
    return sorted(out, key=lambda p: p.name)


def _load_state_ids(worker_dir: Path) -> Set[StateId]:
    state_rows = read_parquet_records(worker_dir / "states.parquet")
    return {str(r.get("state_id", "")) for r in state_rows if str(r.get("state_id", ""))}


def _load_edge_sigs(worker_dir: Path) -> Set[EdgeSig]:
    action_rows = read_parquet_records(worker_dir / "actions.parquet")
    trans_rows = read_parquet_records(worker_dir / "transitions.parquet")

    action_meta: Dict[str, Tuple[str, int]] = {}
    for a in action_rows:
        aid = str(a.get("action_id", ""))
        if not aid:
            continue
        actor = str(a.get("actor", ""))
        idx = int(a.get("canonical_index", -1))
        action_meta[aid] = (actor, idx)

    sigs: Set[EdgeSig] = set()
    for t in trans_rows:
        sid = str(t.get("state_id", ""))
        nsid = str(t.get("next_state_id", ""))
        aid = str(t.get("action_id", ""))
        if not sid or not nsid:
            continue
        actor, idx = action_meta.get(aid, (str(t.get("actor", "")), -1))
        sigs.add((sid, actor, int(idx), nsid))
    return sigs


def _load_sample_group_sigs(worker_dir: Path) -> Set[str]:
    """
    LP sample-group signature:
      one signature per parent state_id that has outgoing transitions,
      representing (state_id + sorted outgoing canonical edges).
    """
    action_rows = read_parquet_records(worker_dir / "actions.parquet")
    trans_rows = read_parquet_records(worker_dir / "transitions.parquet")

    action_meta: Dict[str, Tuple[str, int]] = {}
    for a in action_rows:
        aid = str(a.get("action_id", ""))
        if not aid:
            continue
        actor = str(a.get("actor", ""))
        idx = int(a.get("canonical_index", -1))
        action_meta[aid] = (actor, idx)

    by_state: Dict[str, List[Tuple[str, int, str]]] = defaultdict(list)
    for t in trans_rows:
        sid = str(t.get("state_id", ""))
        nsid = str(t.get("next_state_id", ""))
        aid = str(t.get("action_id", ""))
        if not sid or not nsid:
            continue
        actor, idx = action_meta.get(aid, (str(t.get("actor", "")), -1))
        by_state[sid].append((actor, int(idx), nsid))

    out: Set[str] = set()
    for sid, edges in by_state.items():
        # deterministic canonical serialization so set overlap is meaningful
        edges_sorted = sorted(edges, key=lambda x: (x[0], x[1], x[2]))
        sig_obj = {
            "state_id": sid,
            "edges": edges_sorted,
        }
        out.add(json.dumps(sig_obj, separators=(",", ":"), ensure_ascii=False))
    return out


def _pairwise_overlap(s: Set[str], t: Set[str]) -> Dict[str, float]:
    inter = len(s & t)
    union = len(s | t)
    jacc = (float(inter) / float(union)) if union > 0 else 0.0
    return {
        "intersection": float(inter),
        "union": float(union),
        "jaccard": jacc,
    }


def _pairwise_overlap_edges(s: Set[EdgeSig], t: Set[EdgeSig]) -> Dict[str, float]:
    inter = len(s & t)
    union = len(s | t)
    jacc = (float(inter) / float(union)) if union > 0 else 0.0
    return {
        "intersection": float(inter),
        "union": float(union),
        "jaccard": jacc,
    }


def _owners_map(items_by_worker: Dict[WorkerId, Iterable[str]]) -> Dict[str, List[WorkerId]]:
    owners: Dict[str, List[WorkerId]] = defaultdict(list)
    for worker, items in items_by_worker.items():
        for item in items:
            owners[item].append(worker)
    return owners


def _report_global_duplicates(owners: Dict[str, List[WorkerId]]) -> Dict[str, object]:
    dup_items = {k: v for k, v in owners.items() if len(v) > 1}
    max_owners = max((len(v) for v in dup_items.values()), default=0)
    return {
        "total_unique": int(len(owners)),
        "overlap_unique": int(len(dup_items)),
        "overlap_ratio": float(len(dup_items) / len(owners)) if owners else 0.0,
        "max_worker_owners_for_single_item": int(max_owners),
    }


def _build_round_dir(args: argparse.Namespace) -> Path:
    if args.round_dir:
        return Path(args.round_dir)
    if not args.out_dir:
        raise ValueError("Provide either --round-dir or --out-dir (with --round-idx).")
    return Path(args.out_dir) / f"round_{int(args.round_idx):03d}"


def _summarize_pairwise(data: Sequence[WorkerOverlapData]) -> Dict[str, object]:
    state_pairs: List[Dict[str, object]] = []
    edge_pairs: List[Dict[str, object]] = []
    sample_group_pairs: List[Dict[str, object]] = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            a = data[i]
            b = data[j]
            ov_states = _pairwise_overlap(a.state_ids, b.state_ids)
            ov_edges = _pairwise_overlap_edges(a.edge_sigs, b.edge_sigs)
            ov_sample_groups = _pairwise_overlap(a.sample_group_sigs, b.sample_group_sigs)
            state_pairs.append(
                {
                    "worker_a": a.worker,
                    "worker_b": b.worker,
                    **ov_states,
                }
            )
            edge_pairs.append(
                {
                    "worker_a": a.worker,
                    "worker_b": b.worker,
                    **ov_edges,
                }
            )
            sample_group_pairs.append(
                {
                    "worker_a": a.worker,
                    "worker_b": b.worker,
                    **ov_sample_groups,
                }
            )
    return {
        "state_pairs": state_pairs,
        "edge_pairs": edge_pairs,
        "sample_group_pairs": sample_group_pairs,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Check cross-worker overlap in linear sampler shard tables.")
    ap.add_argument("--round-dir", type=str, default="", help="Path to round dir (e.g. .../linear_sampler/round_000).")
    ap.add_argument("--out-dir", type=str, default="", help="Sampler output dir (e.g. .../linear_sampler).")
    ap.add_argument("--round-idx", type=int, default=0, help="Round index used with --out-dir.")
    ap.add_argument("--top-k", type=int, default=10, help="How many overlapped IDs to print as examples.")
    ap.add_argument("--json-out", type=str, default="", help="Optional path to write full JSON summary.")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    round_dir = _build_round_dir(args)
    worker_dirs = _worker_dirs(round_dir)

    worker_data: List[WorkerOverlapData] = []
    for wd in worker_dirs:
        worker_name = wd.name
        state_ids = _load_state_ids(wd)
        edge_sigs = _load_edge_sigs(wd)
        sample_group_sigs = _load_sample_group_sigs(wd)
        worker_data.append(
            WorkerOverlapData(
                worker=worker_name,
                state_ids=state_ids,
                edge_sigs=edge_sigs,
                sample_group_sigs=sample_group_sigs,
            )
        )

    state_owners = _owners_map({w.worker: w.state_ids for w in worker_data})
    edge_owners = _owners_map(
        {
            w.worker: {json.dumps(list(sig), separators=(",", ":"), ensure_ascii=False) for sig in w.edge_sigs}
            for w in worker_data
        }
    )
    sample_group_owners = _owners_map({w.worker: w.sample_group_sigs for w in worker_data})

    state_global = _report_global_duplicates(state_owners)
    edge_global = _report_global_duplicates(edge_owners)
    sample_group_global = _report_global_duplicates(sample_group_owners)
    pairwise = _summarize_pairwise(worker_data)

    top_state_overlaps = [
        {"state_id": sid, "workers": owners}
        for sid, owners in sorted(
            ((k, v) for k, v in state_owners.items() if len(v) > 1),
            key=lambda kv: (-len(kv[1]), kv[0]),
        )[: int(args.top_k)]
    ]
    top_edge_overlaps = [
        {"edge_sig": json.loads(sig), "workers": owners}
        for sig, owners in sorted(
            ((k, v) for k, v in edge_owners.items() if len(v) > 1),
            key=lambda kv: (-len(kv[1]), kv[0]),
        )[: int(args.top_k)]
    ]
    top_sample_group_overlaps = [
        {"sample_group_sig": json.loads(sig), "workers": owners}
        for sig, owners in sorted(
            ((k, v) for k, v in sample_group_owners.items() if len(v) > 1),
            key=lambda kv: (-len(kv[1]), kv[0]),
        )[: int(args.top_k)]
    ]

    summary = {
        "round_dir": str(round_dir),
        "workers": [w.worker for w in worker_data],
        "per_worker_counts": {
            w.worker: {
                "states": int(len(w.state_ids)),
                "edge_sigs": int(len(w.edge_sigs)),
                "sample_groups": int(len(w.sample_group_sigs)),
            }
            for w in worker_data
        },
        "global_overlap": {
            "states": state_global,
            "edge_sigs": edge_global,
            "sample_groups": sample_group_global,
        },
        "pairwise_overlap": pairwise,
        "examples": {
            "top_state_overlaps": top_state_overlaps,
            "top_edge_overlaps": top_edge_overlaps,
            "top_sample_group_overlaps": top_sample_group_overlaps,
        },
    }

    print(f"[overlap] round_dir={round_dir}")
    print(f"[overlap] workers={', '.join(summary['workers'])}")
    print("[overlap] per-worker counts:")
    for worker, counts in summary["per_worker_counts"].items():
        print(
            f"  - {worker}: "
            f"states={counts['states']} "
            f"edge_sigs={counts['edge_sigs']} "
            f"sample_groups={counts['sample_groups']}"
        )

    s = summary["global_overlap"]["states"]
    e = summary["global_overlap"]["edge_sigs"]
    g = summary["global_overlap"]["sample_groups"]
    print(
        "[overlap] global states: "
        f"overlap_unique={s['overlap_unique']}/{s['total_unique']} "
        f"({100.0 * s['overlap_ratio']:.2f}%), max_owners={s['max_worker_owners_for_single_item']}"
    )
    print(
        "[overlap] global edges: "
        f"overlap_unique={e['overlap_unique']}/{e['total_unique']} "
        f"({100.0 * e['overlap_ratio']:.2f}%), max_owners={e['max_worker_owners_for_single_item']}"
    )
    print(
        "[overlap] global sample_groups: "
        f"overlap_unique={g['overlap_unique']}/{g['total_unique']} "
        f"({100.0 * g['overlap_ratio']:.2f}%), max_owners={g['max_worker_owners_for_single_item']}"
    )

    if summary["pairwise_overlap"]["state_pairs"]:
        print("[overlap] pairwise state overlap:")
        for row in summary["pairwise_overlap"]["state_pairs"]:
            print(
                "  - "
                f"{row['worker_a']} vs {row['worker_b']}: "
                f"intersection={int(row['intersection'])}, "
                f"jaccard={row['jaccard']:.4f}"
            )

    if int(args.top_k) > 0:
        print(f"[overlap] top-{int(args.top_k)} overlapped state IDs:")
        for item in top_state_overlaps:
            print(f"  - state_id={item['state_id']} workers={item['workers']}")
        print(f"[overlap] top-{int(args.top_k)} overlapped sample_groups:")
        for item in top_sample_group_overlaps:
            sg = item["sample_group_sig"]
            print(
                "  - "
                f"state_id={sg.get('state_id')} "
                f"n_edges={len(sg.get('edges', []))} "
                f"workers={item['workers']}"
            )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[overlap] wrote JSON summary: {out_path}")


if __name__ == "__main__":
    main()

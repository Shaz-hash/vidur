from __future__ import annotations
import csv
import re
from pathlib import Path
from typing import Dict, Any, Tuple
from typing import Any

_GAME_RE = re.compile(r"^game_(\d+)_adv_(candidate|best)_ctrl_(candidate|best)\.csv$")

def _f(row: Dict[str, str], key: str, d: float = 0.0) -> float:
    try: return float(row.get(key, d))
    except Exception: return d

def _end_row(path: Path) -> Dict[str, str]:
    end = None
    last = None
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            last = row
            if str(row.get("phase", "")).strip() == "arena_end":
                end = row
    return end or (last or {})

def grade_arena_from_game_logs(
    *,
    game_log_dir: Path,
    out_csv: Path,
    tie_points: float = 0.5,
    win_threshold: float = 0.55,
    eps: float = 1e-9,
) -> Dict[str, Any]:
    per_game: Dict[int, Dict[str, Any]] = {}

    for p in sorted(game_log_dir.glob("game_*_adv_*_ctrl_*.csv")):
        m = _GAME_RE.match(p.name)
        if not m:
            continue
        gid = int(m.group(1))
        adv = m.group(2)
        cycle = "candidate_as_adversary" if adv == "candidate" else "best_as_adversary"

        row = _end_row(p)
        cost = _f(row, "total_cost", _f(row, "slo_violations", 0.0) + _f(row, "total_lateness", 0.0))
        per_game.setdefault(gid, {})[cycle] = {"cost": float(cost), "path": str(p)}

    rows = []
    cand_pts = 0.0
    best_pts = 0.0

    for gid in sorted(per_game.keys()):
        g = per_game[gid]
        if "candidate_as_adversary" not in g or "best_as_adversary" not in g:
            continue
        ca = float(g["candidate_as_adversary"]["cost"])
        cb = float(g["best_as_adversary"]["cost"])

        if ca > cb + eps:
            cp, bp, winner = 1.0, 0.0, "candidate"
        elif cb > ca + eps:
            cp, bp, winner = 0.0, 1.0, "best"
        else:
            cp, bp, winner = float(tie_points), float(tie_points), "tie"

        cand_pts += cp
        best_pts += bp
        rows.append({
            "game_id": gid,
            "candidate_as_adv_cost": ca,
            "best_as_adv_cost": cb,
            "winner": winner,
            "candidate_points": cp,
            "best_points": bp,
            "candidate_as_adv_file": g["candidate_as_adversary"]["path"],
            "best_as_adv_file": g["best_as_adversary"]["path"],
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "game_id", "candidate_as_adv_cost", "best_as_adv_cost",
        "winner", "candidate_points", "best_points",
        "candidate_as_adv_file", "best_as_adv_file",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    total = cand_pts + best_pts
    win_rate = (cand_pts / total) if total > 0 else 0.0
    return {
        "num_eval_roots": len(rows),
        "candidate_points": float(cand_pts),
        "best_points": float(best_pts),
        "total_points": float(total),
        "candidate_win_rate": float(win_rate),
        "arena_win_threshold": float(win_threshold),
        "passed": bool(win_rate > float(win_threshold)),
    }

def extract_model_state(obj: dict[str, Any]) -> dict[str, Any]:
    if isinstance(obj, dict) and "model_state" in obj:
        return obj["model_state"]
    if isinstance(obj, dict):
        return obj
    raise TypeError("Unsupported checkpoint format for model state dict")


class NoopReplayWriter:
    def add(self, _sample: Any) -> None:
        return

    def close(self) -> None:
        return


def write_arena_cycle_end_csv(
    path: Path,
    *,
    game_id: int,
    cycle_label: str,
    total_cost: float,
    slo_violations: int,
    total_lateness: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "game_id",
        "cycle_label",
        "phase",
        "total_cost",
        "slo_violations",
        "total_lateness",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(
            {
                "game_id": int(game_id),
                "cycle_label": str(cycle_label),
                "phase": "arena_end",
                "total_cost": float(total_cost),
                "slo_violations": int(slo_violations),
                "total_lateness": float(total_lateness),
            }
        )
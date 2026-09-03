"""Compare controller action choices in paired candidate/baseline arena blocks."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


PREFILL_RE = re.compile(r"prefill_allocations=(\{[^}]*\})")
HEURISTIC_RE = re.compile(r"heuristic=([^,)]*)")
STRATEGY_RE = re.compile(r"strategy='([^']*)'")


def _prefill_tokens(action: str) -> int:
    match = PREFILL_RE.search(action)
    if not match:
        return 0
    try:
        values = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return 0
    return sum(max(0, int(value)) for value in values.values())


def _field(pattern: re.Pattern[str], action: str, default: str) -> str:
    match = pattern.search(action)
    return match.group(1) if match else default


def analyze(block: Path) -> dict[str, object]:
    prefill = Counter()
    heuristic = Counter()
    strategy = Counter()
    action_rows = 0
    game_costs: list[float] = []
    for path in sorted((block / "arena_games").glob("*model_ctrl_depth1.csv")):
        final_cost = 0.0
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("phase") != "arena_step":
                    continue
                final_cost = float(row.get("total_cost") or final_cost)
                if row.get("player_acted") != "controller":
                    continue
                action = str(row.get("action_repr") or "")
                tokens = _prefill_tokens(action)
                prefill[str(tokens)] += 1
                heuristic[_field(HEURISTIC_RE, action, "none")] += 1
                strategy[_field(STRATEGY_RE, action, "none")] += 1
                action_rows += 1
        game_costs.append(final_cost)
    return {
        "games": len(game_costs),
        "controller_actions": action_rows,
        "mean_final_cost": float(np.mean(game_costs)) if game_costs else None,
        "prefill_tokens_count": dict(sorted(prefill.items(), key=lambda item: int(item[0]))),
        "prefill_tokens_pct": {
            key: 100.0 * value / max(1, action_rows)
            for key, value in sorted(prefill.items(), key=lambda item: int(item[0]))
        },
        "heuristic_pct": {
            key: 100.0 * value / max(1, action_rows) for key, value in heuristic.most_common()
        },
        "strategy_pct": {
            key: 100.0 * value / max(1, action_rows) for key, value in strategy.most_common()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dir", type=Path)
    args = parser.parse_args()
    blocks = {
        "promoted_v_baseline": args.eval_dir
        / "promoted_adversary_vs_promoted_controller_for_controller",
        "candidate": args.eval_dir / "promoted_adversary_vs_candidate_controller",
    }
    print(json.dumps({name: analyze(path) for name, path in blocks.items()}, indent=2))


if __name__ == "__main__":
    main()

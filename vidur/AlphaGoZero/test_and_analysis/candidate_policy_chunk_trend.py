"""Report candidate-controller policy-prior trends by prefill chunk size."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASELINE_BLOCK = "promoted_adversary_vs_promoted_controller_for_controller"
CANDIDATE_BLOCK = "promoted_adversary_vs_candidate_controller"
PREFILL_RE = re.compile(r"prefill_allocations=(\{[^}]*\})")
CHUNKS = (0, 128, 256, 512, 1024, 1536, 2048, 3072, 4096)


def _structured(value: Any, default: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return default
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(value)
        except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
            pass
    return default


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _prefill_tokens(action: Any) -> int:
    match = PREFILL_RE.search(str(action))
    if not match:
        return 0
    values = _structured(match.group(1), {})
    if not isinstance(values, dict):
        return 0
    return sum(max(0, int(_finite(value))) for value in values.values())


def _has_pending_prefill(row: dict[str, Any]) -> bool:
    remaining = _structured(row.get("prefill_remaining_by_id"), {})
    return isinstance(remaining, dict) and any(_finite(value) > 0 for value in remaining.values())


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def analyze_block(block: Path) -> dict[str, Any]:
    rows = 0
    chosen = Counter()
    available = Counter()
    top_prior = Counter()
    prior_mass: dict[int, list[float]] = defaultdict(list)
    best_prior: dict[int, list[float]] = defaultdict(list)

    game_files = sorted((block / "arena_games").glob("*model_ctrl_depth1.csv"))
    for path in game_files:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("player_acted") != "controller" or not _has_pending_prefill(row):
                    continue
                actions = list(_structured(row.get("candidate_top5_action_reprs"), []))
                priors = list(_structured(row.get("candidate_top5_priors"), []))
                count = min(len(actions), len(priors))
                if count <= 0:
                    continue
                actions = actions[:count]
                priors = [max(0.0, _finite(value)) for value in priors[:count]]
                categories = [_prefill_tokens(action) for action in actions]
                rows += 1
                chosen[_prefill_tokens(row.get("action_repr"))] += 1
                top_prior[categories[max(range(count), key=priors.__getitem__)]] += 1
                total = sum(priors)
                for chunk in CHUNKS:
                    indices = [index for index, category in enumerate(categories) if category == chunk]
                    if indices:
                        available[chunk] += 1
                        best_prior[chunk].append(max(priors[index] for index in indices))
                    prior_mass[chunk].append(
                        sum(priors[index] for index in indices) / total if total else 0.0
                    )

    result: dict[str, Any] = {"game_files": len(game_files), "pending_rows": rows}
    for chunk in CHUNKS:
        prefix = "decode" if chunk == 0 else f"prefill_{chunk}"
        result[f"{prefix}_chosen_pct"] = 100.0 * chosen[chunk] / rows if rows else math.nan
        result[f"{prefix}_available_pct"] = 100.0 * available[chunk] / rows if rows else math.nan
        result[f"{prefix}_top_prior_pct"] = 100.0 * top_prior[chunk] / rows if rows else math.nan
        result[f"{prefix}_top5_prior_mass_pct"] = 100.0 * _mean(prior_mass[chunk])
        result[f"{prefix}_best_prior_when_available"] = _mean(best_prior[chunk])
    return result


def _manifest(root: Path, version: int) -> dict[str, Any]:
    path = root / "models" / f"Model_Version{version}" / "candidate_manifest.json"
    return _structured(path.read_text(encoding="utf-8"), {}) if path.is_file() else {}


def _versions(root: Path, start: int, end: int | None) -> list[int]:
    versions = []
    for path in (root / "eval_of_models").glob("eval_*"):
        suffix = path.name.removeprefix("eval_")
        if suffix.isdigit() and int(suffix) >= start and (end is None or int(suffix) <= end):
            versions.append(int(suffix))
    return sorted(set(versions))


def analyze(root: Path, start: int, end: int | None) -> list[dict[str, Any]]:
    output = []
    for version in _versions(root, start, end):
        eval_dir = root / "eval_of_models" / f"eval_{version:06d}"
        manifest = _manifest(root, version)
        baseline = analyze_block(eval_dir / BASELINE_BLOCK)
        candidate = analyze_block(eval_dir / CANDIDATE_BLOCK)
        if not candidate["game_files"]:
            continue
        row: dict[str, Any] = {
            "candidate_version": version,
            "controller_wins": int(_finite(manifest.get("testing_controller_wins"), -1)),
            "controller_promoted": bool(manifest.get("candidate_controller_promoted", False)),
            "baseline_controller_version": int(
                _finite(manifest.get("controller_promoted_selfplay_model_version"), -1)
            ),
            "training_parent_controller_version": int(
                _finite(manifest.get("controller_parent_model_version"), -1)
            ),
        }
        for prefix, summary in (("baseline", baseline), ("candidate", candidate)):
            row.update({f"{prefix}_{key}": value for key, value in summary.items()})
        for chunk in (0, 256, 512):
            name = "decode" if chunk == 0 else f"prefill_{chunk}"
            for metric in ("chosen_pct", "top_prior_pct", "top5_prior_mass_pct"):
                key = f"{name}_{metric}"
                row[f"delta_{key}"] = candidate[key] - baseline[key]
        output.append(row)
    return output


def _fmt(value: Any) -> str:
    number = _finite(value, math.nan)
    return f"{number:.1f}" if math.isfinite(number) else "-"


def print_table(rows: list[dict[str, Any]]) -> None:
    print("cand wins prom pending  topP256 mass256 topP512 mass512 topPdec massDec  dM256 dM512 dMdec")
    for row in rows:
        print(
            f"{row['candidate_version']:>4} {row['controller_wins']:>4} "
            f"{str(row['controller_promoted']):>5} {row['candidate_pending_rows']:>7} "
            f"{_fmt(row['candidate_prefill_256_top_prior_pct']):>8} "
            f"{_fmt(row['candidate_prefill_256_top5_prior_mass_pct']):>7} "
            f"{_fmt(row['candidate_prefill_512_top_prior_pct']):>8} "
            f"{_fmt(row['candidate_prefill_512_top5_prior_mass_pct']):>7} "
            f"{_fmt(row['candidate_decode_top_prior_pct']):>7} "
            f"{_fmt(row['candidate_decode_top5_prior_mass_pct']):>7} "
            f"{_fmt(row['delta_prefill_256_top5_prior_mass_pct']):>6} "
            f"{_fmt(row['delta_prefill_512_top5_prior_mass_pct']):>6} "
            f"{_fmt(row['delta_decode_top5_prior_mass_pct']):>5}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--start-version", type=int, default=100)
    parser.add_argument("--end-version", type=int)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    rows = analyze(args.experiment_root.resolve(), args.start_version, args.end_version)
    print_table(rows)
    if args.csv and rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()

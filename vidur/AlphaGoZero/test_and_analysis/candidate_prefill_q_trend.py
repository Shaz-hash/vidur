"""Report controller prefill-versus-decode trends across AGZ evaluations.

Only controller decisions with at least one positive entry in
``prefill_remaining_by_id`` are eligible.  This deliberately does not use
``canonical_action_count`` as a proxy, because decode-only states can still
have multiple canonical actions.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


BASELINE_BLOCK = "promoted_adversary_vs_promoted_controller_for_controller"
CANDIDATE_BLOCK = "promoted_adversary_vs_candidate_controller"


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _structured(value: Any, default: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return default
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(value)
        except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
            pass
    return default


def has_pending_prefill(row: dict[str, Any]) -> bool:
    remaining = _structured(row.get("prefill_remaining_by_id"), {})
    return isinstance(remaining, dict) and any(
        _finite_float(value) > 0.0 for value in remaining.values()
    )


def action_has_prefill(action_repr: Any) -> bool:
    text = str(action_repr)
    return "prefill_allocations={" in text and "prefill_allocations={}" not in text


def _rate(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return math.nan
    return 100.0 * float(numerator) / float(denominator)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else math.nan


@dataclass
class BlockAccumulator:
    game_files: int = 0
    controller_rows: int = 0
    no_pending_rows: int = 0
    no_pending_single_action_rows: int = 0
    pending_rows: int = 0
    pending_single_action_rows: int = 0
    chosen_prefill: int = 0
    candidate_rows_available: int = 0
    top_visit_prefill: int = 0
    top_q_prefill: int = 0
    top_prior_prefill: int = 0
    both_categories: int = 0
    q_favors_prefill: int = 0
    prior_favors_prefill: int = 0
    q_near_tie: int = 0
    all_top5_prefill: int = 0
    all_top5_decode: int = 0
    pending_decode_choices: int = 0
    decode_choice_q_favors_prefill: int = 0
    decode_choice_prior_favors_prefill: int = 0
    q_margins: list[float] = field(default_factory=list)
    prior_margins: list[float] = field(default_factory=list)
    prefill_visit_mass: list[float] = field(default_factory=list)
    prefill_prior_mass: list[float] = field(default_factory=list)

    def add(self, row: dict[str, Any]) -> None:
        if row.get("player_acted") != "controller":
            return
        self.controller_rows += 1
        if not has_pending_prefill(row):
            self.no_pending_rows += 1
            if _integer(row.get("canonical_action_count")) <= 1:
                self.no_pending_single_action_rows += 1
            return

        self.pending_rows += 1
        if _integer(row.get("canonical_action_count")) <= 1:
            self.pending_single_action_rows += 1
        chosen_is_prefill = action_has_prefill(row.get("action_repr"))
        self.chosen_prefill += int(chosen_is_prefill)
        self.pending_decode_choices += int(not chosen_is_prefill)

        actions = list(_structured(row.get("candidate_top5_action_reprs"), []))
        q_values = list(_structured(row.get("candidate_top5_q_values"), []))
        visits = list(_structured(row.get("candidate_top5_visits"), []))
        priors = list(_structured(row.get("candidate_top5_priors"), []))
        count = min(len(actions), len(q_values), len(visits), len(priors))
        if count <= 0:
            return

        actions = actions[:count]
        q_values = [_finite_float(value, -math.inf) for value in q_values[:count]]
        visits = [max(0, _integer(value)) for value in visits[:count]]
        priors = [max(0.0, _finite_float(value)) for value in priors[:count]]
        prefill = [index for index, action in enumerate(actions) if action_has_prefill(action)]
        decode = [index for index in range(count) if index not in prefill]
        self.candidate_rows_available += 1

        self.top_visit_prefill += int(action_has_prefill(actions[max(range(count), key=visits.__getitem__)]))
        self.top_q_prefill += int(action_has_prefill(actions[max(range(count), key=q_values.__getitem__)]))
        self.top_prior_prefill += int(action_has_prefill(actions[max(range(count), key=priors.__getitem__)]))

        visit_total = sum(visits)
        prior_total = sum(priors)
        self.prefill_visit_mass.append(
            sum(visits[index] for index in prefill) / visit_total if visit_total else 0.0
        )
        self.prefill_prior_mass.append(
            sum(priors[index] for index in prefill) / prior_total if prior_total else 0.0
        )

        if prefill and not decode:
            self.all_top5_prefill += 1
            return
        if decode and not prefill:
            self.all_top5_decode += 1
            return
        if not prefill or not decode:
            return

        self.both_categories += 1
        q_margin = max(q_values[index] for index in prefill) - max(
            q_values[index] for index in decode
        )
        prior_margin = max(priors[index] for index in prefill) - max(
            priors[index] for index in decode
        )
        self.q_margins.append(q_margin)
        self.prior_margins.append(prior_margin)
        q_prefill = q_margin > 0.0
        prior_prefill = prior_margin > 0.0
        self.q_favors_prefill += int(q_prefill)
        self.prior_favors_prefill += int(prior_prefill)
        self.q_near_tie += int(abs(q_margin) <= 0.02)
        if not chosen_is_prefill:
            self.decode_choice_q_favors_prefill += int(q_prefill)
            self.decode_choice_prior_favors_prefill += int(prior_prefill)

    def summary(self) -> dict[str, Any]:
        return {
            "game_files": self.game_files,
            "controller_rows": self.controller_rows,
            "no_pending_rows": self.no_pending_rows,
            "no_pending_single_action_rows": self.no_pending_single_action_rows,
            "pending_prefill_rows": self.pending_rows,
            "pending_single_action_rows": self.pending_single_action_rows,
            "chosen_prefill_pct": _rate(self.chosen_prefill, self.pending_rows),
            "top_visit_prefill_pct": _rate(self.top_visit_prefill, self.candidate_rows_available),
            "top_q_prefill_pct": _rate(self.top_q_prefill, self.candidate_rows_available),
            "top_prior_prefill_pct": _rate(self.top_prior_prefill, self.candidate_rows_available),
            "both_categories_rows": self.both_categories,
            "q_favors_prefill_pct": _rate(self.q_favors_prefill, self.both_categories),
            "prior_favors_prefill_pct": _rate(self.prior_favors_prefill, self.both_categories),
            "q_near_tie_pct": _rate(self.q_near_tie, self.both_categories),
            "q_margin_prefill_minus_decode_mean": _mean(self.q_margins),
            "q_margin_prefill_minus_decode_median": _median(self.q_margins),
            "prior_margin_prefill_minus_decode_mean": _mean(self.prior_margins),
            "prior_margin_prefill_minus_decode_median": _median(self.prior_margins),
            "top5_all_prefill_pct": _rate(self.all_top5_prefill, self.candidate_rows_available),
            "top5_all_decode_pct": _rate(self.all_top5_decode, self.candidate_rows_available),
            "top5_prefill_visit_mass_pct": 100.0 * _mean(self.prefill_visit_mass),
            "top5_prefill_prior_mass_pct": 100.0 * _mean(self.prefill_prior_mass),
            "pending_decode_choices": self.pending_decode_choices,
            "decode_choice_q_favors_prefill_pct": _rate(
                self.decode_choice_q_favors_prefill, self.pending_decode_choices
            ),
            "decode_choice_prior_favors_prefill_pct": _rate(
                self.decode_choice_prior_favors_prefill, self.pending_decode_choices
            ),
        }


def analyze_block(block_dir: Path) -> dict[str, Any]:
    accumulator = BlockAccumulator()
    game_files = sorted((block_dir / "arena_games").glob("*model_ctrl_depth1.csv"))
    accumulator.game_files = len(game_files)
    for game_file in game_files:
        with game_file.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                accumulator.add(row)
    return accumulator.summary()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _versions(eval_root: Path, start: int, end: int | None) -> list[int]:
    found: list[int] = []
    for path in eval_root.glob("eval_*"):
        suffix = path.name.removeprefix("eval_")
        if suffix.isdigit():
            version = int(suffix)
            if version >= start and (end is None or version <= end):
                found.append(version)
    return sorted(set(found))


def _flatten(prefix: str, values: dict[str, Any], output: dict[str, Any]) -> None:
    for key, value in values.items():
        output[f"{prefix}_{key}"] = value


def analyze_version(experiment_root: Path, version: int) -> dict[str, Any]:
    eval_dir = experiment_root / "eval_of_models" / f"eval_{version:06d}"
    manifest = _load_json(
        experiment_root / "models" / f"Model_Version{version}" / "candidate_manifest.json"
    )
    baseline = analyze_block(eval_dir / BASELINE_BLOCK)
    candidate = analyze_block(eval_dir / CANDIDATE_BLOCK)
    status = str(manifest.get("eval_status") or "")
    if not status:
        status = "in_progress" if candidate["game_files"] else "missing"

    row: dict[str, Any] = {
        "candidate_version": version,
        "eval_status": status,
        "baseline_controller_version": _integer(
            manifest.get("controller_promoted_selfplay_model_version"), -1
        ),
        "training_parent_controller_version": _integer(
            manifest.get("controller_parent_model_version"), -1
        ),
        "controller_wins": _integer(manifest.get("testing_controller_wins"), -1),
        "controller_games": _integer(manifest.get("testing_controller_games_compared"), -1),
        "controller_promoted": bool(manifest.get("candidate_controller_promoted", False)),
        "controller_value_rmse": _finite_float(manifest.get("controller_value_rmse"), math.nan),
        "controller_value_p95_abs_error": _finite_float(
            manifest.get("controller_value_p95_abs_error"), math.nan
        ),
    }
    _flatten("baseline", baseline, row)
    _flatten("candidate", candidate, row)
    for metric in (
        "chosen_prefill_pct",
        "top_visit_prefill_pct",
        "top_q_prefill_pct",
        "top_prior_prefill_pct",
        "q_favors_prefill_pct",
        "prior_favors_prefill_pct",
        "q_margin_prefill_minus_decode_mean",
        "q_margin_prefill_minus_decode_median",
        "top5_prefill_visit_mass_pct",
        "top5_prefill_prior_mass_pct",
    ):
        base_value = _finite_float(baseline.get(metric), math.nan)
        candidate_value = _finite_float(candidate.get(metric), math.nan)
        row[f"delta_{metric}"] = (
            candidate_value - base_value
            if math.isfinite(base_value) and math.isfinite(candidate_value)
            else math.nan
        )
    return row


def analyze_experiment(
    experiment_root: Path,
    *,
    start_version: int = 100,
    end_version: int | None = None,
) -> list[dict[str, Any]]:
    eval_root = experiment_root / "eval_of_models"
    return [
        analyze_version(experiment_root, version)
        for version in _versions(eval_root, start_version, end_version)
    ]


def _format(value: Any, digits: int = 1) -> str:
    number = _finite_float(value, math.nan)
    return f"{number:.{digits}f}" if math.isfinite(number) else "-"


def print_table(rows: Iterable[dict[str, Any]]) -> None:
    header = (
        "cand  base parent wins promoted pending  chosenP  Qpref  PriorP "
        "dChosen dQpref dPrior  RMSE status"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        wins = row["controller_wins"]
        games = row["controller_games"]
        wins_text = f"{wins}/{games}" if wins >= 0 and games > 0 else "-"
        print(
            f"{row['candidate_version']:>4} "
            f"{row['baseline_controller_version']:>5} "
            f"{row['training_parent_controller_version']:>6} "
            f"{wins_text:>7} "
            f"{str(bool(row['controller_promoted'])):>8} "
            f"{row['candidate_pending_prefill_rows']:>7} "
            f"{_format(row['candidate_chosen_prefill_pct']):>8} "
            f"{_format(row['candidate_q_favors_prefill_pct']):>6} "
            f"{_format(row['candidate_prior_favors_prefill_pct']):>7} "
            f"{_format(row['delta_chosen_prefill_pct']):>7} "
            f"{_format(row['delta_q_favors_prefill_pct']):>6} "
            f"{_format(row['delta_prior_favors_prefill_pct']):>6} "
            f"{_format(row['controller_value_rmse'], 3):>5} "
            f"{row['eval_status']}"
        )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trend pending-prefill controller Q, policy, visits, and choices across candidates."
    )
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--start-version", type=int, default=100)
    parser.add_argument("--end-version", type=int)
    parser.add_argument("--csv", type=Path, help="Optional detailed CSV output path")
    parser.add_argument("--json", type=Path, help="Optional detailed JSON output path")
    args = parser.parse_args()

    root = args.experiment_root.expanduser().resolve()
    rows = analyze_experiment(
        root,
        start_version=int(args.start_version),
        end_version=args.end_version,
    )
    print_table(rows)
    if args.csv:
        _write_csv(args.csv.expanduser(), rows)
    if args.json:
        args.json.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.json.expanduser().write_text(
            json.dumps(rows, indent=2, sort_keys=True, allow_nan=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

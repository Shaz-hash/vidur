#!/usr/bin/env python3
"""Flatten the real-vLLM scheduler JSONL into a readable action CSV."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Any


ACTION_FIELDS = (
    "token_budget",
    "selected_request_ids",
    "token_allocations",
    "prefill_allocations",
    "decode_allocations",
    "heuristic",
    "strategy",
    "mapping",
    "evicted_request_ids",
)


CSV_FIELDS = (
    "decision_id",
    "mode",
    "policy",
    "action_category",
    "action_label",
    "applied",
    "fallback_reason",
    "sim_time",
    "next_adversary_tick",
    "planning_s",
    "controller_blocking_s",
    "controller_blocking_total_s",
    "live_request_count",
    "live_request_ids",
    "token_budget",
    "selected_internal_request_ids",
    "external_request_ids",
    "allocation_summary",
    "prefill_request_ids",
    "prefill_canonical_tokens",
    "prefill_actual_tokens",
    "decode_request_ids",
    "decode_canonical_tokens",
    "decode_actual_tokens",
    "truncated_request_ids",
    "heuristic",
    "strategy",
    "mapping",
    "token_allocations",
    "prefill_allocations",
    "decode_allocations",
    "evicted_request_ids",
    "terminated_request_ids",
    "action_index",
    "visits",
    "q_value",
    "prior",
    "root_visits",
    "root_value",
    "controller_model_version",
    "adversary_model_version",
    "iterations_requested",
    "state_hash",
    "actual_total_num_scheduled_tokens",
    "actual_num_scheduled_tokens",
    "live_state_fingerprint",
    "plan_state_fingerprint",
    "decision_started_monotonic_s",
    "decision_finished_monotonic_s",
    "action_repr",
)


def _literal_field(action_repr: str, field: str) -> Any:
    patterns = {
        "token_budget": rf"\b{field}=([^,]+)",
        "heuristic": rf"\b{field}=([^,]+)",
        "strategy": rf"\b{field}=([^,]+)",
        "selected_request_ids": rf"\b{field}=(\[[^\]]*\])",
        "evicted_request_ids": rf"\b{field}=(\[[^\]]*\])",
        "token_allocations": rf"\b{field}=(\{{[^}}]*\}})",
        "prefill_allocations": rf"\b{field}=(\{{[^}}]*\}})",
        "decode_allocations": rf"\b{field}=(\{{[^}}]*\}})",
        "mapping": rf"\b{field}=(\([^)]*\))",
    }
    match = re.search(patterns[field], action_repr)
    if not match:
        return None
    try:
        return ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return match.group(1)


def _parse_action(action_repr: str) -> tuple[dict[str, Any], dict[str, Any]]:
    action = {field: _literal_field(action_repr, field) for field in ACTION_FIELDS}
    match = re.search(r"\bselection=(\{.*\})\)$", action_repr)
    selection: dict[str, Any] = {}
    if match:
        try:
            selection = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            pass
    return action, selection


def _json(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _request_ids(allocations: list[dict[str, Any]], phase: str | None = None) -> str:
    return ";".join(
        str(item.get("request_id", ""))
        for item in allocations
        if phase is None or item.get("phase") == phase
    )


def _token_total(
    allocations: list[dict[str, Any]], phase: str, token_field: str
) -> int:
    return sum(
        int(item.get(token_field, 0))
        for item in allocations
        if item.get("phase") == phase
    )


def _action_labels(
    policy: str, prefill_tokens: int, decode_tokens: int
) -> tuple[str, str]:
    if policy == "gv3-real-idle":
        return "idle", "Idle"
    if policy == "gv3-real-decode-fast-forward":
        return "decode_fast_forward", f"Decode fast-forward ({decode_tokens} tokens)"
    if prefill_tokens and decode_tokens:
        return "prefill_and_decode", f"Prefill {prefill_tokens} + decode {decode_tokens}"
    if prefill_tokens:
        return "prefill_only", f"Prefill {prefill_tokens}"
    if decode_tokens:
        return "decode_only", f"Decode only ({decode_tokens} tokens)"
    return "no_allocation", "No allocation"


def _flatten(record: dict[str, Any]) -> dict[str, Any]:
    plan = record.get("plan") or {}
    details = plan.get("details") or {}
    allocations = plan.get("allocations") or []
    action_repr = str(details.get("action_repr") or "")
    action, selection = _parse_action(action_repr)
    policy = str(plan.get("policy") or "")
    prefill_canonical = _token_total(allocations, "prefill", "canonical_tokens")
    prefill_actual = _token_total(allocations, "prefill", "actual_tokens")
    decode_canonical = _token_total(allocations, "decode", "canonical_tokens")
    decode_actual = _token_total(allocations, "decode", "actual_tokens")
    category, label = _action_labels(policy, prefill_canonical, decode_canonical)
    allocation_summary = ";".join(
        f"{item.get('request_id')}:{item.get('phase')}:"
        f"{item.get('actual_tokens')}/{item.get('canonical_tokens')}"
        for item in allocations
    )
    truncated_ids = ";".join(
        str(item.get("request_id"))
        for item in allocations
        if item.get("truncated_to_actual_tail")
    )

    return {
        "decision_id": record.get("decision_id"),
        "mode": record.get("mode"),
        "policy": policy,
        "action_category": category,
        "action_label": label,
        "applied": record.get("applied"),
        "fallback_reason": record.get("fallback_reason"),
        "sim_time": details.get("sim_time"),
        "next_adversary_tick": details.get("next_adversary_tick"),
        "planning_s": record.get("planning_s"),
        "controller_blocking_s": record.get("controller_blocking_s"),
        "controller_blocking_total_s": record.get("controller_blocking_total_s"),
        "live_request_count": len(record.get("live_request_ids") or []),
        "live_request_ids": ";".join(record.get("live_request_ids") or []),
        "token_budget": action.get("token_budget"),
        "selected_internal_request_ids": _json(action.get("selected_request_ids")),
        "external_request_ids": _request_ids(allocations),
        "allocation_summary": allocation_summary,
        "prefill_request_ids": _request_ids(allocations, "prefill"),
        "prefill_canonical_tokens": prefill_canonical,
        "prefill_actual_tokens": prefill_actual,
        "decode_request_ids": _request_ids(allocations, "decode"),
        "decode_canonical_tokens": decode_canonical,
        "decode_actual_tokens": decode_actual,
        "truncated_request_ids": truncated_ids,
        "heuristic": action.get("heuristic"),
        "strategy": action.get("strategy"),
        "mapping": _json(action.get("mapping")),
        "token_allocations": _json(action.get("token_allocations")),
        "prefill_allocations": _json(action.get("prefill_allocations")),
        "decode_allocations": _json(action.get("decode_allocations")),
        "evicted_request_ids": _json(action.get("evicted_request_ids")),
        "terminated_request_ids": _json(details.get("terminated_request_ids")),
        "action_index": selection.get("action_index"),
        "visits": selection.get("visits"),
        "q_value": selection.get("q_value"),
        "prior": selection.get("prior"),
        "root_visits": selection.get("root_visits"),
        "root_value": selection.get("root_value"),
        "controller_model_version": selection.get("controller_model_version"),
        "adversary_model_version": selection.get("adversary_model_version"),
        "iterations_requested": selection.get("iterations_requested"),
        "state_hash": selection.get("state_hash"),
        "actual_total_num_scheduled_tokens": record.get(
            "actual_total_num_scheduled_tokens"
        ),
        "actual_num_scheduled_tokens": _json(
            record.get("actual_num_scheduled_tokens")
        ),
        "live_state_fingerprint": record.get("live_state_fingerprint"),
        "plan_state_fingerprint": plan.get("state_fingerprint"),
        "decision_started_monotonic_s": record.get("decision_started_monotonic_s"),
        "decision_finished_monotonic_s": record.get("decision_finished_monotonic_s"),
        "action_repr": action_repr,
    }


def convert(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.DictWriter(destination, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            writer.writerow(_flatten(record))
            row_count += 1
    return row_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input scheduler JSONL")
    parser.add_argument("output", type=Path, nargs="?", help="Output CSV")
    args = parser.parse_args()
    output = args.output or args.input.with_name(f"{args.input.stem}_actions.csv")
    count = convert(args.input, output)
    print(f"Wrote {count} rows to {output}")


if __name__ == "__main__":
    main()

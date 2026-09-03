from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional


def _j(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        if math.isfinite(x):
            return x
        return default
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _compact_repr(value: Any, max_len: int = 240) -> str:
    text = " ".join(repr(value).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _getattr_float(obj: Any, name: str, default: float = 0.0) -> float:
    return _safe_float(getattr(obj, name, default), default)


def _getattr_int(obj: Any, name: str, default: int = 0) -> int:
    return _safe_int(getattr(obj, name, default), default)


def _finite_or_blank(value: Any) -> Any:
    try:
        x = float(value)
        if math.isfinite(x):
            return x
        return ""
    except Exception:
        return ""


class _CsvWriter:
    def __init__(self, path: str | Path, fieldnames: list[str], *, flush_every: int = 1) -> None:
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.flush_every = max(1, int(flush_every))
        self._file = None
        self._writer: Optional[csv.DictWriter] = None
        self._rows = 0

    def _ensure(self) -> None:
        if self._writer is not None:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames, extrasaction="ignore")
        self._writer.writeheader()

    def write(self, row: Dict[str, Any]) -> None:
        self._ensure()
        assert self._writer is not None

        full = {k: row.get(k, "") for k in self.fieldnames}
        self._writer.writerow(full)
        self._rows += 1

        if self._file is not None and self._rows % self.flush_every == 0:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
        self._file = None
        self._writer = None


class MCTSCsvLogger:
    ROOT_FIELDS = [
        "game_id",
        "root_id",
        "root_node_id",
        "root_depth",
        "root_player",
        "next_player",
        "root_visits",
        "num_valid_actions",
        "num_children",
        "best_action_index",
        "best_action_repr",
        "best_action_value",
        "used_bootstrap",
        "root_mean_value",
        "root_value_sum",
        "root_min_value",
        "root_max_value",
        "lowest_child_value",
        "highest_child_value",
        "action_values_json",
        "valid_mask_json",
        "sim_time",
        "requests_in_system",
        "requests_generated",
        "requests_completed",
        "slo_violations",
        "total_lateness",
        "objective_cost",
        "state_active_ids",
        "state_completed_request_ids",
        "state_dropped_request_ids",
        "state_stopped_decode_request_ids",
        "state_pending_adv_tick",
        "state_decode_credit_balance",
        "state_json",
    ]

    CHILD_FIELDS = [
        "game_id",
        "root_id",
        "root_node_id",
        "parent_node_id",
        "child_node_id",
        "child_depth",
        "parent_player",
        "child_player",
        "action_index",
        "action_repr",
        "reward",
        "parent_time",
        "child_time",
        "transition_discount_time",
        "transition_final_time",
        "discount_time_delta",
        "discount_factor",
        "child_visits",
        "child_value_sum",
        "child_mean_value",
        "child_state_cost",
        "child_min_value",
        "child_max_value",
        "alias_indices_json",
        "child_state_json",
    ]

    def __init__(
        self,
        *,
        root_log_path: str | Path,
        child_log_path: str | Path,
        flush_every: int = 1,
    ) -> None:
        self._root_writer = _CsvWriter(root_log_path, self.ROOT_FIELDS, flush_every=flush_every)
        self._child_writer = _CsvWriter(child_log_path, self.CHILD_FIELDS, flush_every=flush_every)

    def close(self) -> None:
        self._root_writer.close()
        self._child_writer.close()

    def log_root_summary(
        self,
        *,
        game_id: int,
        root_id: int,
        root: Any,
        result: Any,
        root_state: Any,
        state_desc: Optional[Dict[str, Any]] = None,
    ) -> None:
        state_desc = dict(state_desc or {})

        child_values = [
            float(child.mean_value())
            for child in getattr(root, "children", {}).values()
            if _safe_int(getattr(child, "visits", 0)) > 0
        ]

        lowest_child_value = min(child_values) if child_values else ""
        highest_child_value = max(child_values) if child_values else ""

        sim_time = state_desc.get("sim_time", getattr(getattr(root_state, "simulator", None), "_time", 0.0))
        objective_cost = _safe_float(state_desc.get("slo_violations", 0.0)) + _safe_float(
            state_desc.get("total_lateness", 0.0)
        )

        self._root_writer.write(
            {
                "game_id": int(game_id),
                "root_id": int(root_id),
                "root_node_id": _safe_int(getattr(root, "node_id", 0)),
                "root_depth": _safe_int(getattr(root, "depth", 0)),
                "root_player": str(getattr(root, "player", "")),
                "next_player": str(getattr(result, "next_player", "")),
                "root_visits": _safe_int(getattr(root, "visits", 0)),
                "num_valid_actions": sum(1 for x in getattr(root, "valid_mask", []) if bool(x)),
                "num_children": len(getattr(root, "children", {}) or {}),
                "best_action_index": "" if getattr(result, "best_action_index", None) is None else int(result.best_action_index),
                "best_action_repr": _compact_repr(getattr(result, "best_action", "")),
                "best_action_value": _safe_float(getattr(result, "best_action_value", 0.0)),
                "used_bootstrap": bool(getattr(result, "used_bootstrap", False)),
                "root_mean_value": _safe_float(root.mean_value() if hasattr(root, "mean_value") else 0.0),
                "root_value_sum": _safe_float(getattr(root, "value_sum", 0.0)),
                "root_min_value": _finite_or_blank(getattr(root, "min_value", "")),
                "root_max_value": _finite_or_blank(getattr(root, "max_value", "")),
                "lowest_child_value": lowest_child_value,
                "highest_child_value": highest_child_value,
                "action_values_json": _j(getattr(result, "action_values", [])),
                "valid_mask_json": _j(getattr(result, "valid_mask", [])),
                "sim_time": _safe_float(sim_time),
                "requests_in_system": _safe_int(state_desc.get("requests_in_system", 0)),
                "requests_generated": _safe_int(state_desc.get("requests_generated", 0)),
                "requests_completed": _safe_int(state_desc.get("requests_completed", 0)),
                "slo_violations": _safe_int(state_desc.get("slo_violations", 0)),
                "total_lateness": _safe_float(state_desc.get("total_lateness", 0.0)),
                "objective_cost": objective_cost,
                "state_active_ids": _j(state_desc.get("active_request_ids", [])),
                "state_completed_request_ids": _j(state_desc.get("completed_request_ids", [])),
                "state_dropped_request_ids": _j(state_desc.get("dropped_request_ids", [])),
                "state_stopped_decode_request_ids": _j(state_desc.get("stopped_decode_request_ids", [])),
                "state_pending_adv_tick": bool(state_desc.get("pending_adv_tick", False)),
                "state_decode_credit_balance": _safe_int(state_desc.get("decode_credit_balance", 0)),
                "state_json": _j(state_desc),
            }
        )

    def log_root_children(
        self,
        *,
        game_id: int,
        root_id: int,
        root: Any,
        describe_child_state_fn: Optional[Any] = None,
    ) -> None:
        children = getattr(root, "children", {}) or {}

        for action_idx, child in sorted(children.items(), key=lambda x: int(x[0])):
            action = getattr(child, "parent_action", None)
            stats = getattr(child, "cached_stats", None)

            transition_discount_time = getattr(stats, "transition_discount_time", "")
            transition_final_time = getattr(stats, "transition_final_time", "")

            parent_time = _safe_float(getattr(getattr(child, "parent", None), "sim_time", 0.0))
            child_time = _safe_float(getattr(child, "sim_time", 0.0))

            if transition_final_time == "":
                discount_time_delta = child_time - parent_time
            else:
                discount_time_delta = _safe_float(transition_final_time) - parent_time

            aliases = getattr(root, "canonical_to_action_aliases", {}).get(int(action_idx), [int(action_idx)])

            child_state_json = ""
            if describe_child_state_fn is not None:
                try:
                    child_state_json = _j(describe_child_state_fn(child))
                except Exception as exc:
                    child_state_json = _j({"describe_error": repr(exc)})

            self._child_writer.write(
                {
                    "game_id": int(game_id),
                    "root_id": int(root_id),
                    "root_node_id": _safe_int(getattr(root, "node_id", 0)),
                    "parent_node_id": _safe_int(getattr(getattr(child, "parent", None), "node_id", "")),
                    "child_node_id": _safe_int(getattr(child, "node_id", 0)),
                    "child_depth": _safe_int(getattr(child, "depth", 0)),
                    "parent_player": str(getattr(getattr(child, "parent", None), "player", "")),
                    "child_player": str(getattr(child, "player", "")),
                    "action_index": int(action_idx),
                    "action_repr": _compact_repr(action),
                    "reward": _safe_float(getattr(child, "reward", 0.0)),
                    "parent_time": parent_time,
                    "child_time": child_time,
                    "transition_discount_time": "" if transition_discount_time is None else transition_discount_time,
                    "transition_final_time": "" if transition_final_time is None else transition_final_time,
                    "discount_time_delta": _safe_float(discount_time_delta),
                    "discount_factor": _safe_float(getattr(child, "edge_discount", 1.0)),
                    "child_visits": _safe_int(getattr(child, "visits", 0)),
                    "child_value_sum": _safe_float(getattr(child, "value_sum", 0.0)),
                    "child_mean_value": _safe_float(child.mean_value() if hasattr(child, "mean_value") else 0.0),
                    "child_state_cost": _safe_float(getattr(child, "state_cost", 0.0)),
                    "child_min_value": _finite_or_blank(getattr(child, "min_value", "")),
                    "child_max_value": _finite_or_blank(getattr(child, "max_value", "")),
                    "alias_indices_json": _j(aliases),
                    "child_state_json": child_state_json,
                }
            )
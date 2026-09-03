"""Run a canonical request trace against a real vLLM OpenAI endpoint.

Logical arrival time excludes scheduler/controller blocking, matching the
persistent GV3 adapter clock. Raw and controller-time-corrected SLO metrics
are emitted together.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Iterable

from .vllm_live_state import DEFAULT_IMPLICIT_PREFILL_OUTPUT_TOKENS


def _implicit_prefill_output_tokens() -> int:
    value = int(
        os.environ.get(
            "VIDUR_VLLM_GV3_IMPLICIT_PREFILL_OUTPUT_TOKENS",
            str(DEFAULT_IMPLICIT_PREFILL_OUTPUT_TOKENS),
        )
    )
    if value not in {0, 1}:
        raise ValueError(
            "VIDUR_VLLM_GV3_IMPLICIT_PREFILL_OUTPUT_TOKENS must be 0 or 1"
        )
    return value


@dataclass(frozen=True)
class TraceRow:
    request_id: str
    arrived_at_s: float
    actual_prefill_tokens: int
    actual_decode_tokens: int
    actual_prefill_slo_s: float
    actual_decode_slo_s: float
    prompt_ref: str
    seed: int
    ignore_eos: bool


@dataclass(frozen=True)
class RequestObservation:
    row: TraceRow
    submitted_monotonic_s: float
    token_monotonic_s: tuple[float, ...]
    http_completed_monotonic_s: float
    dispatch_lag_s: float
    response_completion_tokens: int | None
    error: str | None
    implicit_prefill_output_tokens: int = (
        DEFAULT_IMPLICIT_PREFILL_OUTPUT_TOKENS
    )


@dataclass(frozen=True)
class BlockingInterval:
    start_s: float
    finish_s: float
    request_ids: frozenset[str]


def _bool(raw: object) -> bool:
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes"}:
        return True
    if value in {"0", "false", "no"}:
        return False
    raise ValueError(f"invalid boolean: {raw!r}")


def load_trace(path: Path) -> list[TraceRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            TraceRow(
                request_id=str(raw["request_id"]),
                arrived_at_s=float(raw["arrived_at_s"]),
                actual_prefill_tokens=int(raw["actual_prefill_tokens"]),
                actual_decode_tokens=int(raw["actual_decode_tokens"]),
                actual_prefill_slo_s=float(raw["actual_prefill_slo_s"]),
                actual_decode_slo_s=float(raw["actual_decode_slo_s"]),
                prompt_ref=str(raw["prompt_ref"]),
                seed=int(raw["seed"]),
                ignore_eos=_bool(raw["ignore_eos"]),
            )
            for raw in csv.DictReader(handle)
        ]
    if not rows:
        raise ValueError(f"empty canonical trace: {path}")
    if len({row.request_id for row in rows}) != len(rows):
        raise ValueError(f"duplicate request IDs in {path}")
    return sorted(rows, key=lambda row: (row.arrived_at_s, row.request_id))


def load_prompt_ids(trace_path: Path, row: TraceRow) -> list[int]:
    prompt_path = (trace_path.parent / row.prompt_ref).resolve()
    document = json.loads(prompt_path.read_text(encoding="utf-8"))
    if str(document.get("request_id")) != row.request_id:
        raise ValueError(f"{row.request_id}: prompt request ID mismatch")
    ids = [int(value) for value in document.get("token_ids", ())]
    if len(ids) != row.actual_prefill_tokens:
        raise ValueError(
            f"{row.request_id}: prompt has {len(ids)} tokens, expected "
            f"{row.actual_prefill_tokens}"
        )
    return ids


def scheduler_blocking_total(path: Path) -> float:
    planning_path = Path(f"{path}.planning")
    if planning_path.is_file():
        try:
            planning = json.loads(planning_path.read_text(encoding="utf-8"))
            started = float(planning["started_monotonic_s"])
            blocking_before = float(planning["blocking_total_before_s"])
            return max(
                0.0,
                blocking_before + max(0.0, time.monotonic() - started),
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            pass
    if not path.is_file():
        return 0.0
    last = ""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = line
    if not last:
        return 0.0
    return max(
        0.0,
        float(json.loads(last).get("controller_blocking_total_s", 0.0)),
    )


def load_blocking_intervals(path: Path) -> list[BlockingInterval]:
    if not path.is_file():
        return []
    intervals: list[BlockingInterval] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            finish = float(row.get("decision_finished_monotonic_s", 0.0))
            start = float(row.get("decision_started_monotonic_s", finish))
            request_ids = frozenset(
                str(value) for value in row.get("live_request_ids", ())
            )
            if finish >= start and request_ids:
                intervals.append(BlockingInterval(start, finish, request_ids))
    return intervals


def blocked_between(
    intervals: Iterable[BlockingInterval],
    request_id: str,
    start_s: float,
    finish_s: float,
) -> float:
    if finish_s <= start_s:
        return 0.0
    total = 0.0
    for interval in intervals:
        if request_id not in interval.request_ids:
            continue
        total += max(
            0.0,
            min(finish_s, interval.finish_s) - max(start_s, interval.start_s),
        )
    return min(finish_s - start_s, total)


async def _wait_for_logical_arrival(
    *,
    start_s: float,
    arrival_s: float,
    scheduler_log: Path,
) -> float:
    loop = asyncio.get_running_loop()
    while True:
        now = loop.time()
        logical_elapsed = (
            now - start_s - scheduler_blocking_total(scheduler_log)
        )
        remaining = arrival_s - logical_elapsed
        if remaining <= 0.0:
            return max(0.0, -remaining)
        await asyncio.sleep(min(0.02, remaining))


async def _execute_request(
    *,
    session: Any,
    base_url: str,
    model: str,
    trace_path: Path,
    row: TraceRow,
    benchmark_start_s: float,
    scheduler_log: Path,
    timeout_s: float,
) -> RequestObservation:
    dispatch_lag = await _wait_for_logical_arrival(
        start_s=benchmark_start_s,
        arrival_s=row.arrived_at_s,
        scheduler_log=scheduler_log,
    )
    submitted = time.monotonic()
    token_times: list[float] = []
    completion_tokens: int | None = None
    error: str | None = None
    output_offset = _implicit_prefill_output_tokens()
    physical_output_tokens = row.actual_decode_tokens + output_offset
    payload = {
        "model": model,
        "prompt": load_prompt_ids(trace_path, row),
        "max_tokens": physical_output_tokens,
        "temperature": 0.0,
        "seed": row.seed,
        "ignore_eos": row.ignore_eos,
        "stream": True,
        "stream_options": {"include_usage": True},
        "logprobs": 1,
    }
    try:
        async with session.post(
            f"{base_url.rstrip('/')}/v1/completions",
            json=payload,
            headers={"X-Request-Id": row.request_id},
            timeout=timeout_s,
        ) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {body[:500]}")
            while True:
                raw_line = await response.content.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                usage = event.get("usage")
                if (
                    isinstance(usage, dict)
                    and usage.get("completion_tokens") is not None
                ):
                    completion_tokens = int(usage["completion_tokens"])
                for choice in event.get("choices", ()):
                    logprobs = choice.get("logprobs") or {}
                    tokens = logprobs.get("tokens") or ()
                    observed = time.monotonic()
                    token_times.extend(observed for _ in tokens)
                # The engine has finished this request once every expected token
                # is observed. Do not wait indefinitely for an optional trailing
                # usage/[DONE] event after scheduler completion.
                if len(token_times) >= physical_output_tokens:
                    break

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    completed = time.monotonic()
    if error is None:
        expected = physical_output_tokens
        if completion_tokens is not None and completion_tokens != expected:
            error = (
                f"completion token count {completion_tokens} != "
                f"expected {expected}"
            )
        elif len(token_times) != expected:
            error = (
                f"stream token count {len(token_times)} != expected {expected}"
            )
    return RequestObservation(
        row=row,
        submitted_monotonic_s=submitted,
        token_monotonic_s=tuple(token_times),
        http_completed_monotonic_s=completed,
        dispatch_lag_s=dispatch_lag,
        response_completion_tokens=completion_tokens,
        error=error,
        implicit_prefill_output_tokens=output_offset,
    )


def request_metrics(
    observation: RequestObservation,
    intervals: list[BlockingInterval],
) -> dict[str, object]:
    row = observation.row
    submitted = observation.submitted_monotonic_s
    tokens = observation.token_monotonic_s
    output_offset = int(observation.implicit_prefill_output_tokens)
    if output_offset not in {0, 1}:
        raise ValueError("implicit prefill output offset must be 0 or 1")
    if not tokens:
        return {
            **asdict(row),
            "completed": False,
            "error": observation.error or "no output tokens",
            "dispatch_lag_s": observation.dispatch_lag_s,
        }

    raw_ttft = tokens[0] - submitted
    blocking_to_first = blocked_between(
        intervals, row.request_id, submitted, tokens[0]
    )
    corrected_ttft = max(0.0, raw_ttft - blocking_to_first)
    raw_decode_lateness = 0.0
    corrected_decode_lateness = 0.0
    raw_intervals: list[float] = []
    corrected_intervals: list[float] = []
    for previous, current in zip(tokens, tokens[1:]):
        raw_delta = max(0.0, current - previous)
        blocked = blocked_between(
            intervals, row.request_id, previous, current
        )
        corrected_delta = max(0.0, raw_delta - blocked)
        raw_intervals.append(raw_delta)
        corrected_intervals.append(corrected_delta)
        raw_decode_lateness += max(
            0.0, raw_delta - row.actual_decode_slo_s
        )
        corrected_decode_lateness += max(
            0.0, corrected_delta - row.actual_decode_slo_s
        )

    raw_prefill_lateness = max(
        0.0, raw_ttft - row.actual_prefill_slo_s
    )
    corrected_prefill_lateness = max(
        0.0, corrected_ttft - row.actual_prefill_slo_s
    )
    raw_total_lateness = raw_prefill_lateness + raw_decode_lateness
    corrected_total_lateness = (
        corrected_prefill_lateness + corrected_decode_lateness
    )
    raw_completion = observation.http_completed_monotonic_s - submitted
    total_blocking = blocked_between(
        intervals,
        row.request_id,
        submitted,
        observation.http_completed_monotonic_s,
    )
    return {
        **asdict(row),
        "completed": observation.error is None,
        "error": observation.error or "",
        "dispatch_lag_s": observation.dispatch_lag_s,
        "tokens_observed": max(0, len(tokens) - output_offset),
        "physical_output_tokens_observed": len(tokens),
        "implicit_prefill_output_tokens": output_offset,
        "raw_ttft_s": raw_ttft,
        "corrected_ttft_s": corrected_ttft,
        "raw_mean_inter_token_s": (
            statistics.mean(raw_intervals) if raw_intervals else 0.0
        ),
        "corrected_mean_inter_token_s": (
            statistics.mean(corrected_intervals)
            if corrected_intervals
            else 0.0
        ),
        "raw_prefill_lateness_s": raw_prefill_lateness,
        "corrected_prefill_lateness_s": corrected_prefill_lateness,
        "raw_decode_lateness_s": raw_decode_lateness,
        "corrected_decode_lateness_s": corrected_decode_lateness,
        "raw_total_lateness_s": raw_total_lateness,
        "corrected_total_lateness_s": corrected_total_lateness,
        "raw_slo_violation": int(raw_total_lateness > 0.0),
        "corrected_slo_violation": int(corrected_total_lateness > 0.0),
        "raw_completion_s": raw_completion,
        "corrected_completion_s": max(
            0.0, raw_completion - total_blocking
        ),
        "controller_blocking_s": total_blocking,
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(
    *,
    policy: str,
    request_rows: list[dict[str, object]],
    scheduler_log: Path,
    wall_seconds: float,
) -> dict[str, object]:
    completed = [
        row for row in request_rows if bool(row.get("completed"))
    ]
    failures = [
        row for row in request_rows if not bool(row.get("completed"))
    ]
    raw_lateness = sum(
        float(row.get("raw_total_lateness_s", 0.0)) for row in completed
    )
    corrected_lateness = sum(
        float(row.get("corrected_total_lateness_s", 0.0))
        for row in completed
    )
    raw_violations = sum(
        int(row.get("raw_slo_violation", 0)) for row in completed
    )
    corrected_violations = sum(
        int(row.get("corrected_slo_violation", 0)) for row in completed
    )
    raw_ttft = [float(row["raw_ttft_s"]) for row in completed]
    corrected_ttft = [
        float(row["corrected_ttft_s"]) for row in completed
    ]
    return {
        "policy": policy,
        "request_count": len(request_rows),
        "completed_requests": len(completed),
        "failed_requests": len(failures),
        "wall_seconds": wall_seconds,
        "scheduler_blocking_total_s": scheduler_blocking_total(
            scheduler_log
        ),
        "raw_slo_violations": raw_violations,
        "raw_total_lateness_s": raw_lateness,
        "raw_total_cost": raw_violations + raw_lateness,
        "corrected_slo_violations": corrected_violations,
        "corrected_total_lateness_s": corrected_lateness,
        "corrected_total_cost": (
            corrected_violations + corrected_lateness
        ),
        "raw_mean_ttft_s": (
            statistics.mean(raw_ttft) if raw_ttft else 0.0
        ),
        "raw_p95_ttft_s": _percentile(raw_ttft, 0.95),
        "corrected_mean_ttft_s": (
            statistics.mean(corrected_ttft)
            if corrected_ttft
            else 0.0
        ),
        "corrected_p95_ttft_s": _percentile(
            corrected_ttft, 0.95
        ),
        "errors": [
            str(row.get("error", "")) for row in failures
        ],
    }


async def run(args: argparse.Namespace) -> dict[str, object]:
    import aiohttp

    trace_path = Path(args.trace).expanduser().resolve()
    scheduler_log = Path(args.scheduler_log).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_trace(trace_path)
    if scheduler_log.exists() and scheduler_log.stat().st_size:
        raise RuntimeError(
            f"scheduler log must be empty for a benchmark run: "
            f"{scheduler_log}"
        )

    timeout = aiohttp.ClientTimeout(
        total=None,
        sock_connect=30,
        sock_read=args.timeout_s,
    )
    benchmark_start = time.monotonic()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        observations = await asyncio.gather(
            *(
                _execute_request(
                    session=session,
                    base_url=args.base_url,
                    model=args.model,
                    trace_path=trace_path,
                    row=row,
                    benchmark_start_s=benchmark_start,
                    scheduler_log=scheduler_log,
                    timeout_s=args.timeout_s,
                )
                for row in rows
            )
        )
    wall_seconds = time.monotonic() - benchmark_start
    intervals = load_blocking_intervals(scheduler_log)
    request_rows = [
        request_metrics(item, intervals) for item in observations
    ]
    fieldnames = sorted({key for row in request_rows for key in row})
    with (output_dir / "request_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(request_rows)
    summary = summarize(
        policy=args.policy,
        request_rows=request_rows,
        scheduler_log=scheduler_log,
        wall_seconds=wall_seconds,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["failed_requests"]:
        raise RuntimeError(
            f"{summary['failed_requests']} benchmark requests failed"
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--scheduler-log", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000"
    )
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    return parser.parse_args()


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()

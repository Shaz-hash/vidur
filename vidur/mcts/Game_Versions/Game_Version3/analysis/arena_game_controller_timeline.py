from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ARENA_CSV = (
    "simulator_output/Game_Version3_Fresh8_Hops200/"
    "Model_Tester_Results_50games_5s/arena_games/"
    "game_18000000_model_adv_depth1_vs_model_ctrl_depth1.csv"
)


def _repo_root() -> Path:
    # arena_game_controller_timeline.py -> analysis -> Game_Version3
    # -> Game_Versions -> mcts -> vidur(package) -> repo root.
    return Path(__file__).resolve().parents[5]


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return _repo_root() / p


def _float(row: dict[str, str], *names: str, default: float = 0.0) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    return float(default)


def _parse_structured(value: str) -> Any:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        pass
    try:
        return ast.literal_eval(value)
    except Exception:
        return None


def _prefill_request_count(value: str) -> int:
    parsed = _parse_structured(value)
    if isinstance(parsed, dict):
        count = 0
        for remaining in parsed.values():
            try:
                if float(remaining) > 0.0:
                    count += 1
            except Exception:
                count += 1
        return count
    if isinstance(parsed, (list, tuple, set)):
        return len(parsed)
    return 0


def _load_controller_points(arena_csv: Path, *, time_column: str) -> list[dict[str, Any]]:
    with arena_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    points: list[dict[str, Any]] = []
    for row in rows:
        if row.get("player_acted") != "controller":
            continue
        points.append(
            {
                "game_id": row.get("game_id", ""),
                "turn": int(row.get("turn") or 0),
                "time_sec": _float(row, time_column),
                "abs_chosen_q_value": abs(_float(row, "chosen_q_value", "choosen_q_value")),
                "abs_chosen_reward": abs(_float(row, "chosen_reward", "choosen_reward")),
                "prefill_request_count": _prefill_request_count(row.get("prefill_remaining_by_id", "")),
            }
        )

    points.sort(key=lambda p: (float(p["time_sec"]), int(p["turn"])))
    if not points:
        raise RuntimeError(f"no controller rows found in {arena_csv}")
    return points


def _write_points_csv(path: Path, points: Iterable[dict[str, Any]]) -> None:
    points = list(points)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(points[0].keys()))
        writer.writeheader()
        writer.writerows(points)


def _plot_q_reward(
    path: Path,
    points: list[dict[str, Any]],
    *,
    game_id: str,
    game_duration_sec: float,
) -> None:
    import matplotlib.pyplot as plt

    x = [float(p["time_sec"]) for p in points]
    q = [float(p["abs_chosen_q_value"]) for p in points]
    reward = [float(p["abs_chosen_reward"]) for p in points]

    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    ax.plot(x, q, color="#1f77b4", marker="o", linewidth=2.0, markersize=4, label="|chosen Q value|")
    ax.plot(x, reward, color="#ff7f0e", marker="s", linewidth=2.0, markersize=4, label="|chosen reward|")
    ax.set_xlim(0.0, float(game_duration_sec))
    ax.set_xlabel("Simulator time (seconds)")
    ax.set_ylabel("Controller-perspective absolute value")
    ax.set_title(f"Game {game_id}: controller chosen Q and reward over time")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_q_reward_prefill(
    path: Path,
    points: list[dict[str, Any]],
    *,
    game_id: str,
    game_duration_sec: float,
) -> None:
    import matplotlib.pyplot as plt

    x = [float(p["time_sec"]) for p in points]
    q = [float(p["abs_chosen_q_value"]) for p in points]
    reward = [float(p["abs_chosen_reward"]) for p in points]
    prefill_count = [int(p["prefill_request_count"]) for p in points]

    fig, (ax_value, ax_prefill) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(11.5, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.25]},
    )
    ax_value.plot(
        x,
        q,
        color="#1f77b4",
        marker="o",
        linewidth=2.0,
        markersize=4,
        label="|chosen Q value|",
    )
    ax_value.plot(
        x,
        reward,
        color="#ff7f0e",
        marker="s",
        linewidth=2.0,
        markersize=4,
        label="|chosen reward|",
    )
    ax_value.set_xlim(0.0, float(game_duration_sec))
    ax_value.set_ylabel("Controller-perspective absolute value")
    ax_value.set_title(f"Game {game_id}: controller Q, reward, and prefill pressure")
    ax_value.grid(True, alpha=0.25)
    ax_value.legend(loc="best")

    ax_prefill.plot(
        x,
        prefill_count,
        color="#2ca02c",
        marker="^",
        linewidth=2.0,
        markersize=4,
        label="prefill requests in system",
    )
    ax_prefill.set_xlim(0.0, float(game_duration_sec))
    ax_prefill.set_xlabel("Simulator time (seconds)")
    ax_prefill.set_ylabel("Prefill request count")
    max_prefill = max(prefill_count) if prefill_count else 0
    ax_prefill.set_ylim(-0.25, max(1.0, float(max_prefill) + 0.75))
    ax_prefill.set_yticks(list(range(0, max_prefill + 1)))
    ax_prefill.grid(True, alpha=0.25)
    ax_prefill.legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    arena_csv = _resolve_path(args.arena_csv)
    if not arena_csv.exists():
        raise FileNotFoundError(f"arena CSV not found: {arena_csv}")

    points = _load_controller_points(arena_csv, time_column=args.time_column)
    game_id = str(points[0]["game_id"] or arena_csv.stem.split("_")[1])

    output_dir = _resolve_path(args.output_dir) if args.output_dir else arena_csv.parents[2] / "analysis" / f"arena_game_{game_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    points_csv = output_dir / f"game_{game_id}_controller_timeline_points.csv"
    q_reward_png = output_dir / f"game_{game_id}_controller_q_reward.png"
    q_reward_prefill_png = output_dir / f"game_{game_id}_controller_q_reward_prefill.png"

    _write_points_csv(points_csv, points)
    _plot_q_reward(q_reward_png, points, game_id=game_id, game_duration_sec=float(args.game_duration_sec))
    _plot_q_reward_prefill(
        q_reward_prefill_png,
        points,
        game_id=game_id,
        game_duration_sec=float(args.game_duration_sec),
    )

    print(f"points_csv={points_csv}")
    print(f"q_reward_plot={q_reward_png}")
    print(f"q_reward_prefill_plot={q_reward_prefill_png}")
    print(f"controller_rows={len(points)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot controller selected Q/reward timeline from a GV3 arena game CSV."
    )
    parser.add_argument("--arena-csv", default=DEFAULT_ARENA_CSV)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--game-duration-sec", type=float, default=5.0)
    parser.add_argument(
        "--time-column",
        choices=["sim_time_before", "sim_time_after"],
        default="sim_time_after",
        help=(
            "Timestamp for each controller action. Default uses sim_time_after so the "
            "prefill-count line aligns with the post-action state logged in the row."
        ),
    )
    return parser


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()

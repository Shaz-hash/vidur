"""Dump two-level shared-root MCTS visit distributions for one arena-start state."""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path
from typing import Any

from vidur.bellman_v4_adv.arena_mcts_value_runner2 import (
    _build_env,
    _limit_native_threads,
    _load_model,
    _make_mcts_config,
    _patch_tree_mcts_runtime_issues,
    _valid_canonical_root_actions,
)
from vidur.Game_Version3.mcts import Node, VidurMCTS


def _repr(value: Any, max_len: int = 500) -> str:
    text = " ".join(repr(value).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _node_row(*, game_id: int, hop: int, turn: int, root_child: Node, grandchild: Node | None = None) -> dict[str, Any]:
    child = root_child if grandchild is None else grandchild
    parent = child.parent
    return {
        "game_id": int(game_id),
        "hop": int(hop),
        "turn": int(turn),
        "level": 1 if grandchild is None else 2,
        "root_child_node_id": int(root_child.node_id),
        "root_child_action_index": "" if root_child.parent_action_index is None else int(root_child.parent_action_index),
        "root_child_action_repr": _repr(root_child.parent_action),
        "parent_node_id": "" if parent is None else int(parent.node_id),
        "parent_player": "" if parent is None else str(parent.player),
        "child_node_id": int(child.node_id),
        "child_player": str(child.player),
        "action_index": "" if child.parent_action_index is None else int(child.parent_action_index),
        "action_repr": _repr(child.parent_action),
        "visits": int(child.visits),
        "value_sum": float(child.value_sum),
        "mean_value": float(child.mean_value()),
        "reward": float(child.reward),
        "edge_discount": float(child.edge_discount),
        "state_cost": float(child.state_cost),
        "sim_time": float(child.sim_time),
        "min_value": float(child.min_value) if child.min_value != float("inf") else "",
        "max_value": float(child.max_value) if child.max_value != float("-inf") else "",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "game_id",
        "hop",
        "turn",
        "level",
        "root_child_node_id",
        "root_child_action_index",
        "root_child_action_repr",
        "parent_node_id",
        "parent_player",
        "child_node_id",
        "child_player",
        "action_index",
        "action_repr",
        "visits",
        "value_sum",
        "mean_value",
        "reward",
        "edge_discount",
        "state_cost",
        "sim_time",
        "min_value",
        "max_value",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def run(args: argparse.Namespace) -> Path:
    _limit_native_threads(int(args.worker_threads))
    model_path = Path(args.model_path).expanduser() if args.model_path else None
    model = _load_model(model_path, feature_dim=int(args.feature_dim))
    simulator, env = _build_env(args)
    cfg = _make_mcts_config(args, iterations=int(args.mcts_iterations), seed=int(args.seed))
    tree = VidurMCTS(env=env, mctsConfig=cfg)
    _patch_tree_mcts_runtime_issues(tree)

    try:
        if int(args.hop) != 0:
            raise NotImplementedError("This diagnostic currently supports hop 0 only.")

        state = env.initial_state()
        player, _actions, _mask, canonical_indices = _valid_canonical_root_actions(tree, state, str(args.start_player))
        if not canonical_indices:
            raise RuntimeError(f"No canonical actions for player={player}")

        tree.search_dnn(
            dnn_model=model,
            rootState=state,
            root_player=str(player),
            game_id=int(args.game_id),
            root_id=int(args.root_id),
            root_node_id_override=int(args.root_node_id),
            root_depth=0,
            mcts_iter=int(args.mcts_iterations),
            model_version=int(args.model_version),
            use_model_bootstrap=bool(int(args.model_version) > 0),
            root_phase="two_step_visit_debug",
            cycle_label="hop0_turn0",
            turn=int(args.turn),
        )

        root = tree._root
        if root is None:
            raise RuntimeError("MCTS root missing after search")

        root_rows: list[dict[str, Any]] = []
        grand_rows: list[dict[str, Any]] = []
        for _idx, root_child in sorted(root.children.items(), key=lambda item: int(item[0])):
            root_rows.append(_node_row(game_id=args.game_id, hop=args.hop, turn=args.turn, root_child=root_child))
            for _gidx, grandchild in sorted(root_child.children.items(), key=lambda item: int(item[0])):
                grand_rows.append(
                    _node_row(
                        game_id=args.game_id,
                        hop=args.hop,
                        turn=args.turn,
                        root_child=root_child,
                        grandchild=grandchild,
                    )
                )

        out_dir = Path(args.output_dir).expanduser()
        root_path = out_dir / "root_child_visit_distribution.csv"
        grand_path = out_dir / "grandchild_visit_distribution.csv"
        combined_path = out_dir / "two_step_visit_distribution.csv"
        _write_csv(root_path, root_rows)
        _write_csv(grand_path, grand_rows)
        _write_csv(combined_path, root_rows + grand_rows)
        print(f"[two-step] root_player={root.player} root_visits={root.visits} root_children={len(root_rows)} grandchildren={len(grand_rows)}")
        print(f"[two-step] wrote {root_path}")
        print(f"[two-step] wrote {grand_path}")
        print(f"[two-step] wrote {combined_path}")
        return combined_path
    finally:
        try:
            tree.close()
        except Exception:
            pass
        for method in ("shutdown", "close", "stop"):
            fn = getattr(simulator, method, None)
            if callable(fn):
                try:
                    fn()
                    break
                except Exception:
                    pass
        gc.collect()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log root-child and grandchild MCTS visit distributions for hop0/turn0.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-version", type=int, required=True)
    parser.add_argument("--feature-dim", type=int, default=226)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--game-id", type=int, default=12_000_000)
    parser.add_argument("--hop", type=int, default=0)
    parser.add_argument("--turn", type=int, default=0)
    parser.add_argument("--root-id", type=int, default=12_000_000_000)
    parser.add_argument("--root-node-id", type=int, default=12_000_000_000)
    parser.add_argument("--start-player", choices=("adversary", "controller"), default="adversary")
    parser.add_argument("--mcts-iterations", type=int, default=10_000)
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument("--uct-c", type=float, default=1.4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--discount-factor", type=float, default=0.995)
    parser.add_argument("--discount-time-denom", type=float, default=0.015725797204323228)
    parser.add_argument("--arena-time-limit-sec", type=float, default=5.0)
    parser.add_argument("--environment-lang", choices=("python", "native"), default="python")
    return parser.parse_args()


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()

# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

from __future__ import annotations

import csv
from collections import deque
from typing import Optional


def dump_tree_snapshot_csv(
    *,
    out_path: str,
    game_id: int,
    root_id: int,
    sim_iteration: int,
    root_node,
    minmax_min: float,
    minmax_max: float,
    max_nodes: Optional[int] = None,
) -> None:
    """
    Snapshot the current MCTS tree to CSV.

    Rows repeat across iterations (same node_id appears in many snapshots),
    because each call dumps the whole tree "so far".

    root_node is your MCTSNode object.
    """
    fieldnames = [
        "game_id",
        "root_id",
        "sim_iteration",
        "node_id",
        "parent_node_id",
        "depth",
        "player_acted_to_create_this_node",
        "player_to_act_in_this_node",
        "action_index",
        "action_repr",
        "prior",
        "nn_value_controller",
        "value_sum",
        "visits",
        "children_visits_sum",
        "mean_value",
        "num_children",
        "minmax_minimum",
        "minmax_maximum",
    ]

    seen: set[int] = set()
    q = deque([root_node])

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        while q:
            node = q.popleft()
            nid = int(node.node_id)
            if nid in seen:
                continue
            seen.add(nid)

            parent = node.parent
            parent_id = int(parent.node_id) if parent is not None else None

            children_visits_sum = sum(int(ch.visits) for ch in node.children.values())
            num_children = len(node.children)
            mean_value = float(node.mean_value()) if getattr(node, "visits", 0) else 0.0

            w.writerow(
                {
                    "game_id": int(game_id),
                    "root_id": int(root_id),
                    "sim_iteration": int(sim_iteration),
                    "node_id": nid,
                    "parent_node_id": ("" if parent_id is None else parent_id),
                    "depth": int(node.depth),
                    "player_acted_to_create_this_node": (parent.player if parent is not None else "root_no_parent"),
                    "player_to_act_in_this_node": node.player,
                    "action_index": ("" if node.parent_action_index is None else int(node.parent_action_index)),
                    "action_repr": ("" if node.parent_action is None else repr(node.parent_action)),
                    "prior": float(getattr(node, "prior", 0.0)),
                    "nn_value_controller": ("" if node.nn_value_controller is None else float(node.nn_value_controller)),
                    "value_sum": float(getattr(node, "value_sum", 0.0)),
                    "visits": int(getattr(node, "visits", 0)),
                    "children_visits_sum": int(children_visits_sum),
                    "mean_value": float(mean_value),
                    "num_children": int(num_children),
                    "minmax_minimum": float(minmax_min),
                    "minmax_maximum": float(minmax_max),
                }
            )

            if max_nodes is not None and len(seen) >= int(max_nodes):
                break

            for child in node.children.values():
                q.append(child)

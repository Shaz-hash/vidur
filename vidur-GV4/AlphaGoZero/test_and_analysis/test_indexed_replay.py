"""Correctness checks for the indexed replay sampler."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from vidur.AlphaGoZero.indexed_replay import (
    ADVERSARY_ACTION_DIM,
    CONTROLLER_ACTION_DIM,
    ensure_partition_indexes,
    indexed_sample_training_data,
    materialize_markov_policy_arrays,
    materialize_policy_arrays,
    sample_addresses,
)
from vidur.AlphaGoZero.markov_value_features import (
    GLOBAL_DIM,
    LAUNCH_DIM,
    MARKOV_VALUE_SCHEMA,
    REQUEST_DIM,
)


STATE_FIELDS = [
    "accepted_shard_id",
    "source_worker_id",
    "game_id",
    "turn_number",
    "depth_number",
    "player",
    "feature_complete",
    "state_features_json",
    "target_value",
    "value_feature_complete",
    "value_feature_schema",
    "value_global_features_json",
    "value_request_features_json",
    "value_launch_features_json",
    "value_request_count",
    "value_launch_count",
]
POLICY_FIELDS = [
    "accepted_shard_id",
    "source_worker_id",
    "game_id",
    "turn_number",
    "depth_number",
    "player",
    "canon_action_index",
    "visit_count",
    "action_features_json",
]


def _partition(root: Path, *, role: str, shard: int, rows: int) -> Path:
    partition = root / role / f"shard_{shard:03d}"
    partition.mkdir(parents=True)
    state_path = partition / "replay_target_runtime_feature_complete.csv"
    policy_path = partition / "replay_policy_rows.csv"
    action_dim = CONTROLLER_ACTION_DIM if role == "controller" else ADVERSARY_ACTION_DIM
    with state_path.open("w", encoding="utf-8", newline="") as state_handle, policy_path.open(
        "w", encoding="utf-8", newline=""
    ) as policy_handle:
        state_writer = csv.DictWriter(state_handle, fieldnames=STATE_FIELDS)
        policy_writer = csv.DictWriter(policy_handle, fieldnames=POLICY_FIELDS)
        state_writer.writeheader()
        policy_writer.writeheader()
        for index in range(rows):
            common = {
                "accepted_shard_id": f"{role}_{shard}",
                "source_worker_id": f"worker{shard}",
                "game_id": shard * 10_000 + index,
                "turn_number": index,
                "depth_number": index % 3,
                "player": role,
            }
            request_count = index % 4
            launch_count = index % 3
            state_writer.writerow(
                {
                    **common,
                    "feature_complete": 1,
                    "state_features_json": json.dumps([float(index)] * 226),
                    "target_value": float(-index) / 100.0,
                    "value_feature_complete": 1,
                    "value_feature_schema": MARKOV_VALUE_SCHEMA,
                    "value_global_features_json": json.dumps([float(index)] * GLOBAL_DIM),
                    "value_request_features_json": json.dumps(
                        [[float(index)] * REQUEST_DIM for _ in range(request_count)]
                    ),
                    "value_launch_features_json": json.dumps(
                        [[float(index)] * LAUNCH_DIM for _ in range(launch_count)]
                    ),
                    "value_request_count": request_count,
                    "value_launch_count": launch_count,
                }
            )
            for action_index, visits in ((0, 3), (1, 7)):
                policy_writer.writerow(
                    {
                        **common,
                        "canon_action_index": action_index,
                        "visit_count": visits,
                        "action_features_json": json.dumps(
                            [float(action_index)] * action_dim
                        ),
                    }
                )
    return state_path


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="agz_indexed_replay_") as raw:
        root = Path(raw)
        paths = [
            _partition(root, role=role, shard=shard, rows=25)
            for role in ("controller", "adversary")
            for shard in range(3)
        ]
        descriptors, cache_metrics = ensure_partition_indexes(
            paths,
            value_feature_schema=MARKOV_VALUE_SCHEMA,
            workers=3,
        )
        assert len(descriptors) == 6
        assert int(cache_metrics["indexed_cache_files_rebuilt"]) == 6

        first, total = sample_addresses(
            descriptors,
            role="controller",
            sample_size=40,
            seed=2127,
        )
        repeat, _ = sample_addresses(
            descriptors,
            role="controller",
            sample_size=40,
            seed=2127,
        )
        next_version, _ = sample_addresses(
            descriptors,
            role="controller",
            sample_size=40,
            seed=2128,
        )
        first_ids = [(row.cache_dir, row.local_row) for row in first]
        assert total == 75
        assert first_ids == [(row.cache_dir, row.local_row) for row in repeat]
        assert len(first_ids) == len(set(first_ids)) == 40
        assert first_ids != [(row.cache_dir, row.local_row) for row in next_version]

        sampled, controller_roots, adversary_roots, counts, timings = indexed_sample_training_data(
            paths,
            value_feature_schema=MARKOV_VALUE_SCHEMA,
            seed=2127,
            controller_value_rows=30,
            adversary_value_rows=20,
            controller_policy_roots=18,
            adversary_policy_roots=12,
            cache_workers=3,
            extraction_workers=2,
        )
        assert len(sampled) == 50
        assert sum(row["player"] == "controller" for row in sampled) == 30
        assert sum(row["player"] == "adversary" for row in sampled) == 20
        assert len(controller_roots) == 18
        assert len(adversary_roots) == 12
        assert counts["controller"] == 75 and counts["adversary"] == 75
        assert int(timings["indexed_unique_value_addresses"]) == 50

        (controller_arrays, controller_policy) = materialize_policy_arrays(
            controller_roots,
            action_dim=CONTROLLER_ACTION_DIM,
            root_cap=15,
        )
        X, y, probabilities, offsets = controller_arrays
        assert X.shape == (30, 226 + CONTROLLER_ACTION_DIM)
        assert y.shape == probabilities.shape == (30,)
        assert len(offsets) == int(controller_policy["roots_with_actions"]) == 15
        for begin, end in offsets:
            np.testing.assert_allclose(probabilities[begin:end], [0.3, 0.7], atol=1e-7)
            assert abs(float(np.sum(probabilities[begin:end])) - 1.0) < 1e-7

        (markov_arrays, markov_policy) = materialize_markov_policy_arrays(
            controller_roots,
            action_dim=CONTROLLER_ACTION_DIM,
            root_cap=15,
        )
        states, actions, _target_logits, markov_probabilities, markov_offsets = markov_arrays
        assert len(states) == len(markov_offsets) == 15
        assert actions.shape == (30, CONTROLLER_ACTION_DIM)
        assert markov_probabilities.shape == (30,)
        assert int(markov_policy["state_rows_materialized"]) == 15
        assert int(markov_policy["state_per_action_expansion"]) == 0
        for begin, end in markov_offsets:
            np.testing.assert_allclose(
                markov_probabilities[begin:end], [0.3, 0.7], atol=1e-7
            )

    print(
        json.dumps(
            {
                "ok": True,
                "same_seed_reproducible": True,
                "candidate_seed_changes_selection": True,
                "without_replacement": True,
                "role_uniform_global_sampling": True,
                "policy_alignment": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

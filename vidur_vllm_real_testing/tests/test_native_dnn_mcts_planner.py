from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from vidur_vllm_real_testing.native_dnn_mcts_planner import (
    _action_from_json,
    _native_export_header,
    _require_dnn_artifact,
    PromotedNativeDNNMCTSPlanner,
)


def _request(
    request_id: int,
    *,
    prefill: int,
    prefill_done: int,
    decode: int = 216,
    decode_done: int = 0,
) -> dict[str, object]:
    return {
        "request_id": request_id,
        "num_prefill_tokens": prefill,
        "num_processed_prefill_tokens": prefill_done,
        "num_decode_tokens": decode,
        "num_processed_decode_tokens": decode_done,
        "is_prefill_complete": prefill_done == prefill,
        "completed": False,
        "dropped": False,
    }


class NativeDNNMCTSPlannerTest(unittest.TestCase):
    def test_action_translation_preserves_native_allocations(self) -> None:
        payload = {
            "requests": [
                _request(3, prefill=1024, prefill_done=512),
                _request(7, prefill=256, prefill_done=256, decode_done=9),
            ]
        }
        action = {
            "type": "controller",
            "token_budget": 129,
            "selected_request_ids": [3, 7],
            "evicted_request_ids": [],
            "token_allocations": {"3": 128, "7": 1},
            "prefill_allocations": {"3": 128},
            "decode_allocations": {"7": 1},
            "heuristic": "SJF",
            "strategy": "GV2|evict_none",
            "mapping": [0, 1, 0],
        }
        result = _action_from_json(
            json.dumps(action),
            state_payload=payload,
            selection={"action_index": 4, "visits": 321},
        )
        self.assertEqual(result.token_budget, 129)
        self.assertEqual(result.prefill_allocations, {3: 128})
        self.assertEqual(result.decode_allocations, {7: 1})
        self.assertEqual(result.selection["visits"], 321)

    def test_action_translation_rejects_overallocated_prefill(self) -> None:
        payload = {"requests": [_request(3, prefill=1024, prefill_done=960)]}
        action = {
            "type": "controller",
            "token_budget": 128,
            "selected_request_ids": [3],
            "evicted_request_ids": [],
            "token_allocations": {"3": 128},
            "prefill_allocations": {"3": 128},
            "decode_allocations": {},
            "mapping": [0, 1, 0],
        }
        with self.assertRaisesRegex(ValueError, "over-allocates"):
            _action_from_json(
                json.dumps(action),
                state_payload=payload,
                selection={},
            )

    def test_winner_matches_evaluator_visit_q_index_order(self) -> None:
        native_out = {
            "root_visits": 12,
            "root_value_sum": -18.0,
            "root_action_values": [-3.0, -2.0, -2.0],
            "children": [
                {"index": 0, "visits": 5, "prior": 0.2, "parent_action_json": "a"},
                {"index": 1, "visits": 5, "prior": 0.3, "parent_action_json": "b"},
                {"index": 2, "visits": 2, "prior": 0.5, "parent_action_json": "c"},
            ],
        }
        index, action_json, selection = PromotedNativeDNNMCTSPlanner._winning_action(
            native_out
        )
        self.assertEqual(index, 1)
        self.assertEqual(action_json, "b")
        self.assertEqual(selection["q_value"], -2.0)

    def test_export_header_rejects_hgb(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.tsv"
            path.write_text("hgb226_v1\nmodel_tag\tlegacy\n", encoding="utf-8")
            with self.assertRaisesRegex(TypeError, "not an AlphaGoZero DNN"):
                _native_export_header(path)

    def test_artifact_requires_markov_dnn_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata.json").write_text(
                json.dumps(
                    {
                        "model_kind": "value_dnn",
                        "architecture_version": "agz_markov_value_deepset_v2",
                        "feature_schema": "legacy_226",
                        "role": "controller",
                    }
                ),
                encoding="utf-8",
            )
            (root / "native_model.tsv").write_text(
                "agz_dnn_v2\nmodel_kind\tvalue_dnn\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(TypeError, "feature_schema"):
                _require_dnn_artifact(
                    root,
                    model_kind="value_dnn",
                    architecture="agz_markov_value_deepset_v2",
                    role="controller",
                )


if __name__ == "__main__":
    unittest.main()


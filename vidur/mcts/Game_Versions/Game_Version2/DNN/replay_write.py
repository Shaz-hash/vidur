# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

"""
replay_writer.py

Disk-backed dataset writer for DNN-MCTS training samples (root positions).

- One "sample" == one root position:
  { ids, player, model_inputs, action_mask, mcts_policy_target, mcts_value_target }

- Samples are buffered and written in shards using torch.save():
    replay_000000.pt, replay_000001.pt, ...

- A manifest.jsonl is appended with one JSON line per shard, so later loaders can
  discover shards quickly without scanning the directory.

NOTE: torch.save uses pickle. Only torch.load data you trust.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypedDict

import torch

from .dnn_spec import DEFAULT_DNN_SPEC, num_actions_for_player


def _num_actions(player: str) -> int:
    if player not in ("controller", "adversary"):
        raise ValueError(f"Unknown player={player!r}")
    return int(num_actions_for_player(player, spec=DEFAULT_DNN_SPEC))


class RootSample(TypedDict):
    feature_version: int

    game_id: int
    root_id: int
    root_node_id: int
    root_depth: int
    player: str  # "controller" | "adversary"

    # inputs (CPU tensors, no batch dimension)
    inputs: Dict[str, Any]  # req_features/global_features/req_mask/action_mask

    # targets
    targets: Dict[str, Any]  # policy/value

    # optional debug info
    meta: Dict[str, Any]


def _jsonl_append(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _to_cpu(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.detach().to("cpu")


def _strip_batch(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None
    if t.dim() >= 1 and t.shape[0] == 1:
        return t[0]
    return t

def pack_model_inputs_for_storage(inputs: Any) -> Dict[str, Any]:
    prefill_req_features = _strip_batch(_to_cpu(getattr(inputs, "prefill_req_features", None)))
    decode_req_features = _strip_batch(_to_cpu(getattr(inputs, "decode_req_features", None)))
    global_features = _strip_batch(_to_cpu(getattr(inputs, "global_features", None)))
    prefill_req_mask = _strip_batch(_to_cpu(getattr(inputs, "prefill_req_mask", None)))
    decode_req_mask = _strip_batch(_to_cpu(getattr(inputs, "decode_req_mask", None)))
    action_mask = _strip_batch(_to_cpu(getattr(inputs, "action_mask", None)))

    req_features = _strip_batch(_to_cpu(getattr(inputs, "req_features", None)))
    req_mask = _strip_batch(_to_cpu(getattr(inputs, "req_mask", None)))

    if global_features is None:
        raise ValueError("ModelInputs.global_features is required")

    # Prefer split tensors; allow legacy fallback if still present.
    if prefill_req_features is None or decode_req_features is None:
        if req_features is None:
            raise ValueError("ModelInputs must contain split req features or legacy req_features")
        n_p = int(DEFAULT_DNN_SPEC.n_prefill_req)
        n_d = int(DEFAULT_DNN_SPEC.n_decode_req)
        prefill_req_features = req_features[:n_p]
        decode_req_features = req_features[n_p:n_p + n_d]

    out = {
        "prefill_req_features": prefill_req_features,
        "decode_req_features": decode_req_features,
        "global_features": global_features,
        "prefill_req_mask": prefill_req_mask,
        "decode_req_mask": decode_req_mask,
        "action_mask": action_mask,
        "req_features": req_features,  # optional legacy
        "req_mask": req_mask,          # optional legacy
    }
    return out


def make_root_sample(
    *,
    feature_version: int,
    game_id: int,
    root_id: int,
    root_node_id: int,
    root_depth: int,
    player: str,
    model_inputs: Any,
    action_mask: Sequence[bool],
    mcts_policy: Sequence[float],
    mcts_value_controller: float,
    meta: Optional[Dict[str, Any]] = None,
) -> RootSample:
    
    expected_a = _num_actions(player)
    if len(action_mask) != expected_a:
        raise ValueError(
            f"action_mask length {len(action_mask)} != expected {expected_a} for player={player}"
        )
    if len(mcts_policy) != expected_a:
        raise ValueError(
            f"mcts_policy length {len(mcts_policy)} != expected {expected_a} for player={player}"
        )

  
    packed_inputs = pack_model_inputs_for_storage(model_inputs)

    # overwrite/ensure action_mask in stored inputs is consistent with env mask
    packed_inputs["action_mask"] = torch.tensor(action_mask, dtype=torch.bool)

    targets = {
        "policy": torch.tensor(mcts_policy, dtype=torch.float32),
        "value": torch.tensor(float(mcts_value_controller), dtype=torch.float32),
    }

    return {
        "feature_version": int(feature_version),
        "game_id": int(game_id),
        "root_id": int(root_id),
        "root_node_id": int(root_node_id),
        "root_depth": int(root_depth),
        "player": str(player),
        "inputs": packed_inputs,
        "targets": targets,
        "meta": meta or {},
    }


@dataclass
class ReplayWriterConfig:
    out_dir: Path
    shard_size: int = 512
    prefix: str = "replay"
    manifest_name: str = "manifest.jsonl"


class ReplayWriter:
    """
    Buffers RootSample records and flushes to disk as torch.save(list[RootSample]).
    Also appends a manifest.jsonl entry per shard.
    """

    def __init__(self, cfg: ReplayWriterConfig) -> None:
        self.cfg = cfg
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cfg.out_dir / self.cfg.manifest_name

        self._buffer: List[RootSample] = []
        self._shard_index = self._infer_next_shard_index()

    def _infer_next_shard_index(self) -> int:
        # Simple approach: look for existing replay_*.pt files.
        existing = sorted(self.cfg.out_dir.glob(f"{self.cfg.prefix}_*.pt"))
        if not existing:
            return 0
        # replay_000123.pt -> 123
        last = existing[-1].stem
        try:
            idx = int(last.split("_")[-1])
            return idx + 1
        except Exception:
            return len(existing)

    def add(self, sample: RootSample) -> None:
        self._buffer.append(sample)
        if len(self._buffer) >= int(self.cfg.shard_size):
            self.flush()

    def flush(self) -> Optional[Path]:
        if not self._buffer:
            return None

        shard_path = self.cfg.out_dir / f"{self.cfg.prefix}_{self._shard_index:06d}.pt"
        tmp_path = shard_path.with_suffix(".pt.tmp")

        # torch.save list[RootSample] (dicts + tensors) atomically
        torch.save(self._buffer, tmp_path)
        tmp_path.replace(shard_path)
        # TODO: we might not need min, max game ids here
        # manifest entry
        game_ids = [s["game_id"] for s in self._buffer]
        root_ids = [s["root_id"] for s in self._buffer]
        entry = {
            "time": time.time(),
            "path": str(shard_path),
            "shard_index": self._shard_index,
            "num_samples": len(self._buffer),
            "feature_version": int(self._buffer[0]["feature_version"]),
            "min_game_id": int(min(game_ids)),
            "max_game_id": int(max(game_ids)),
            "min_root_id": int(min(root_ids)),
            "max_root_id": int(max(root_ids)),
        }
        _jsonl_append(self.manifest_path, entry)

        # reset buffer
        self._buffer = []
        self._shard_index += 1
        return shard_path

    def close(self) -> None:
        self.flush()

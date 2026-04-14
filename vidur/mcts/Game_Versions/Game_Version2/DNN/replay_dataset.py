# (بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيْمِ)

"""
replay_dataset.py

Loads DNN-MCTS root samples saved by replay_write.py.

- Reads shard list from manifest.jsonl
- Loads shards via torch.load(map_location="cpu") with LRU caching
- Provides sample_batch(batch_size) across shards
- Collates samples into stacked tensors (split by player)

Security note: torch.load uses pickle. Only load data you trust.
"""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .dnn_spec import DEFAULT_DNN_SPEC, num_actions_for_player
from .replay_write import RootSample

def _num_actions(player: str) -> int:
    if player not in ("controller", "adversary"):
        raise ValueError(f"Unknown player={player!r}")
    return int(num_actions_for_player(player, spec=DEFAULT_DNN_SPEC))

## TODO : We can move this into types.py later

@dataclass(frozen=True)
class ManifestEntry:
    path: Path
    shard_index: int
    num_samples: int
    feature_version: int


def load_manifest(manifest_path: Path) -> List[ManifestEntry]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    entries: List[ManifestEntry] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            entries.append(
                ManifestEntry(
                    path=Path(obj["path"]),
                    shard_index=int(obj.get("shard_index", -1)),
                    num_samples=int(obj["num_samples"]),
                    feature_version=int(obj.get("feature_version", 1)),
                )
            )

    if not entries:
        raise ValueError(f"manifest is empty: {manifest_path}")
    return entries


class ReplayShardSampler:
    """
    Samples RootSample records from disk shards.

    - Uses manifest.jsonl to discover shard files
    - Loads shards lazily
    - Keeps a small LRU cache of loaded shards
    """

    def __init__(
        self,
        dataset_dir: Path,
        *,
        max_cached_shards: int = 2,
        seed: int = 0,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.manifest_path = dataset_dir / "manifest.jsonl"
        self.entries = load_manifest(self.manifest_path)

        self._rng = random.Random(seed)
        self._max_cached = int(max_cached_shards)

        # LRU cache: path -> (samples_list, indices_by_player)
        self._cache: "OrderedDict[Path, Tuple[List[RootSample], Dict[str, List[int]]]]" = OrderedDict()

        # weights for shard selection
        self._weights = [e.num_samples for e in self.entries]
        self._total = sum(self._weights)
        if self._total <= 0:
            raise ValueError("manifest has non-positive total samples")

    def _load_shard(self, path: Path) -> Tuple[List[RootSample], Dict[str, List[int]]]:
        # LRU hit
        if path in self._cache:
            val = self._cache.pop(path)
            self._cache[path] = val
            return val

        samples = torch.load(path, map_location="cpu")
        if not isinstance(samples, list):
            raise TypeError(f"Shard must contain a list, got {type(samples)} at {path}")

        indices_by_player: Dict[str, List[int]] = {"controller": [], "adversary": []}
        for i, s in enumerate(samples):
            p = s.get("player")
            if p in indices_by_player:
                indices_by_player[p].append(i)

        val = (samples, indices_by_player)
        self._cache[path] = val

        # evict LRU
        while len(self._cache) > self._max_cached:
            self._cache.popitem(last=False)

        return val

    def _pick_shard_entry(self) -> ManifestEntry:
        # random.choices is fine for small lists
        return self._rng.choices(self.entries, weights=self._weights, k=1)[0]

    def sample_one(self, *, player: Optional[str] = None) -> RootSample:
        """
        If player is set, rejection-samples shards until it finds one that has that player.
        """
        if player is not None and player not in ("controller", "adversary"):
            raise ValueError(f"Unknown player={player!r}")

        for _ in range(200):  # avoid infinite loop if dataset lacks player
            entry = self._pick_shard_entry()
            shard, idxs = self._load_shard(entry.path)

            if not shard:
                continue

            if player is None:
                i = self._rng.randrange(len(shard))
                return shard[i]

            candidate = idxs.get(player, [])
            if not candidate:
                continue
            i = self._rng.choice(candidate)
            return shard[i]

        raise RuntimeError(f"Could not sample player={player!r} from dataset={self.dataset_dir}")

    def sample_batch(self, batch_size: int, *, player: Optional[str] = None) -> List[RootSample]:
        return [self.sample_one(player=player) for _ in range(int(batch_size))]


def _stack_or_default_bool(
    items: List[Optional[torch.Tensor]],
    *,
    shape: Tuple[int, int],
) -> torch.Tensor:
    """
    items: list of 1D bool tensors or None
    returns: [B, N] bool
    """
    bsz = len(items)
    out = torch.ones((bsz, shape[1]), dtype=torch.bool)
    for i, t in enumerate(items):
        if t is None:
            continue
        out[i] = t.to(dtype=torch.bool)
    return out


def split_by_player(samples: Sequence[RootSample]) -> Dict[str, List[RootSample]]:
    out: Dict[str, List[RootSample]] = {"controller": [], "adversary": []}
    for s in samples:
        p = s["player"]
        if p in out:
            out[p].append(s)
    return out


def _stack_bool_optional(items: List[Optional[torch.Tensor]], width: int) -> Optional[torch.Tensor]:
    if not any(x is not None for x in items):
        return None
    out = torch.zeros((len(items), width), dtype=torch.bool)
    for i, t in enumerate(items):
        if t is None:
            continue
        out[i] = t.to(dtype=torch.bool)
    return out


def collate_player_samples(
    samples: Sequence[RootSample],
    *,
    device: torch.device,
) -> Dict[str, Any]:
    if not samples:
        raise ValueError("No samples to collate")

    player = samples[0]["player"]
    if any(s["player"] != player for s in samples):
        raise ValueError("collate_player_samples received mixed players")

    a = _num_actions(player)

    n_p = int(DEFAULT_DNN_SPEC.n_prefill_req)
    n_d = int(DEFAULT_DNN_SPEC.n_decode_req)

    prefill_req_features = torch.stack([s["inputs"]["prefill_req_features"] for s in samples], dim=0).to(device)
    decode_req_features = torch.stack([s["inputs"]["decode_req_features"] for s in samples], dim=0).to(device)
    global_features = torch.stack([s["inputs"]["global_features"] for s in samples], dim=0).to(device)

    prefill_req_mask = _stack_bool_optional(
        [s["inputs"].get("prefill_req_mask") for s in samples], n_p
    )
    decode_req_mask = _stack_bool_optional(
        [s["inputs"].get("decode_req_mask") for s in samples], n_d
    )
    if prefill_req_mask is not None:
        prefill_req_mask = prefill_req_mask.to(device)
    if decode_req_mask is not None:
        decode_req_mask = decode_req_mask.to(device)

    action_mask = torch.stack([s["inputs"]["action_mask"] for s in samples], dim=0).to(device).to(torch.bool)
    if action_mask.shape[1] != a:
        raise ValueError(f"action_mask width {action_mask.shape[1]} != expected {a} for player={player}")

    target_policy = torch.stack([s["targets"]["policy"] for s in samples], dim=0).to(device)
    target_value = torch.stack([s["targets"]["value"] for s in samples], dim=0).to(device).view(-1)

    target_policy = target_policy * action_mask.to(dtype=target_policy.dtype)
    denom = target_policy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    target_policy = target_policy / denom

    # optional legacy fields for compatibility
    req_features = torch.cat([prefill_req_features, decode_req_features], dim=1)
    req_mask = None
    if prefill_req_mask is not None and decode_req_mask is not None:
        req_mask = torch.cat([prefill_req_mask, decode_req_mask], dim=1)

    return {
        "player": player,
        "prefill_req_features": prefill_req_features,
        "decode_req_features": decode_req_features,
        "global_features": global_features,
        "prefill_req_mask": prefill_req_mask,
        "decode_req_mask": decode_req_mask,
        "req_features": req_features,  # legacy
        "req_mask": req_mask,          # legacy
        "action_mask": action_mask,
        "target_policy": target_policy,
        "target_value": target_value,
        "ids": {
            "game_id": torch.tensor([s["game_id"] for s in samples], device=device),
            "root_id": torch.tensor([s["root_id"] for s in samples], device=device),
            "root_node_id": torch.tensor([s["root_node_id"] for s in samples], device=device),
            "root_depth": torch.tensor([s["root_depth"] for s in samples], device=device),
        },
    }




def collate_mixed_samples(
    samples: Sequence[RootSample],
    *,
    device: torch.device,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Returns:
      {"controller": batch_or_None, "adversary": batch_or_None}
    """
    groups = split_by_player(samples)
    out: Dict[str, Optional[Dict[str, Any]]] = {"controller": None, "adversary": None}
    for p in ["controller", "adversary"]:
        if groups[p]:
            out[p] = collate_player_samples(groups[p], device=device)
    return out

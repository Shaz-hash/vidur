from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class LoadedWeights:
    feature_names: List[str]
    weights: np.ndarray


def _load_feature_names_json(path: Path) -> List[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = payload.get("feature_names")
    if not isinstance(names, list) or not names:
        raise ValueError(f"invalid feature names file: {path}")
    out = [str(x) for x in names]
    if any(not x for x in out):
        raise ValueError(f"empty feature name in: {path}")
    return out


def _extract_weights_mapping(payload: Any) -> Dict[str, float]:
    if isinstance(payload, dict) and isinstance(payload.get("weights_by_feature"), dict):
        src = payload["weights_by_feature"]
    elif isinstance(payload, dict):
        src = payload
    else:
        raise ValueError("weights json must be an object")
    out: Dict[str, float] = {}
    for k, v in src.items():
        out[str(k)] = float(v)
    if not out:
        raise ValueError("weights json has no feature weights")
    return out


def load_weights_from_lp_solution_dir(solution_dir: str) -> LoadedWeights:
    d = Path(solution_dir)
    npz_path = d / "lp_solution.npz"
    names_path = d / "lp_feature_names.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing lp solution file: {npz_path}")
    if not names_path.exists():
        raise FileNotFoundError(f"missing feature names file: {names_path}")

    names = _load_feature_names_json(names_path)
    arr = np.load(npz_path)
    if "weights" not in arr:
        raise ValueError(f"'weights' key missing in: {npz_path}")
    weights = np.asarray(arr["weights"], dtype=np.float64).reshape(-1)

    if len(names) != int(weights.shape[0]):
        raise ValueError(
            f"feature count mismatch: names={len(names)} weights={int(weights.shape[0])}"
        )
    return LoadedWeights(feature_names=names, weights=weights)


def load_weights_from_json(
    weights_json: str,
    *,
    feature_names_json: Optional[str] = None,
) -> LoadedWeights:
    w_path = Path(weights_json)
    payload = json.loads(w_path.read_text(encoding="utf-8"))
    mapping = _extract_weights_mapping(payload)

    names_path: Optional[Path] = None
    if feature_names_json:
        names_path = Path(feature_names_json)
    else:
        sibling = w_path.parent / "lp_feature_names.json"
        if sibling.exists():
            names_path = sibling

    if names_path is not None:
        if not names_path.exists():
            raise FileNotFoundError(f"feature names file not found: {names_path}")
        names = _load_feature_names_json(names_path)
    else:
        names = sorted(mapping.keys())

    missing = [n for n in names if n not in mapping]
    if missing:
        raise ValueError(f"weights json missing features: {missing[:5]}")

    weights = np.asarray([float(mapping[n]) for n in names], dtype=np.float64)
    return LoadedWeights(feature_names=names, weights=weights)


def load_weights(
    *,
    lp_solution_dir: Optional[str],
    weights_json: Optional[str],
    feature_names_json: Optional[str],
) -> LoadedWeights:
    has_dir = bool(str(lp_solution_dir or "").strip())
    has_json = bool(str(weights_json or "").strip())
    if has_dir == has_json:
        raise ValueError("provide exactly one of --lp-solution-dir or --weights-json")

    if has_dir:
        return load_weights_from_lp_solution_dir(str(lp_solution_dir))
    return load_weights_from_json(str(weights_json), feature_names_json=feature_names_json)


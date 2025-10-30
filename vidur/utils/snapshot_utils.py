from __future__ import annotations

import copy
from typing import Any


# def clone_mutable(value: Any) -> Any:
#     """Return a lightweight recursive copy of common mutable containers.

#     We avoid ``copy.deepcopy`` to give callers fine-grained control over which
#     objects get duplicated while still preventing aliasing on basic Python
#     collections.
#     """

#     if isinstance(value, list):
#         return [clone_mutable(item) for item in value]
#     if isinstance(value, tuple):
#         # Preserve tuple / namedtuple types when possible.
#         if hasattr(value, "_fields"):  # NamedTuple
#             return type(value)(*(clone_mutable(item) for item in value))
#         return tuple(clone_mutable(item) for item in value)
#     if isinstance(value, dict):
#         return {key: clone_mutable(val) for key, val in value.items()}
#     if isinstance(value, set):
#         return {clone_mutable(item) for item in value}
#     if isinstance(value, frozenset):
#         return frozenset(clone_mutable(item) for item in value)
#     if hasattr(value, "copy"):
#         try:
#             return value.copy()
#         except TypeError:
#             pass
#     try:
#         return copy.deepcopy(value)
#     except Exception:
#         return value


# snapshot_utils.py
from typing import Any

_PRIM = (int, float, bool, str, type(None))

def to_primitive_tree(x: Any, *, allow_sets: bool = True) -> Any:
    """Return a JSON-friendly snapshot tree (lists/dicts of primitives).
    - tuples/namedtuples -> lists
    - sets/frozensets -> sorted lists (stable)
    - dict keys must be str or int
    Raises TypeError on unsupported objects.
    """
    if isinstance(x, _PRIM):
        return x

    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if not isinstance(k, (str, int)):
                raise TypeError(f"Non-primitive dict key: {type(k).__name__}")
            out[k] = to_primitive_tree(v, allow_sets=allow_sets)
        return out

    if isinstance(x, (list, tuple)):
        return [to_primitive_tree(v, allow_sets=allow_sets) for v in x]

    if allow_sets and isinstance(x, (set, frozenset)):
        # Sort for determinism; elements must be primitives after recursion
        lst = [to_primitive_tree(v, allow_sets=allow_sets) for v in x]
        return sorted(lst, key=lambda z: (str(type(z)), str(z)))

    # Explicit whitelist any other types you need (e.g., numpy RNG state dicts)
    raise TypeError(f"Unsupported type in snapshot: {type(x).__name__}")




# --- Back-compat shim for older code ---
def clone_mutable(value: Any) -> Any:
    """
    Compatibility shim to keep older code running.
    We DO NOT deep-copy arbitrary objects; we normalize to primitives where possible.
    If normalization fails, we just return the value (best-effort).
    """
    try:
        return to_primitive_tree(value)
    except Exception:
        # Last-resort fallback: return as-is instead of deepcopying heavy objects
        return value







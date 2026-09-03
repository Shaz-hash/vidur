"""Per-(state, action) features parsed from a ControllerAction object.

These describe what the action *does* at the parent state, without simulating
its effect. Combined with cliff/forecast features for the parent, they let a
Q(s, a) regressor distinguish "this action saves the bottleneck" from
"this action does nothing" without relying on outcome features (which would
leak the target).
"""
from __future__ import annotations
from typing import Any


# Eviction rule names in canonical order (matches GV3 ControllerActionConfig).
EVICT_RULES = (
    "evict_none",
    "evict_largest_prefill",
    "evict_earliest_prefill_deadline",
    "evict_prefill_missed_deadline",
    "evict_prefill_lateness_over_0p5",
    "evict_longest_decode",
    "evict_decode_lateness_over_0p5",
    "evict_prefill_highest_lateness",
    "evict_decode_highest_lateness",
)
ORDERING_HEURISTICS = ("SJF", "EDF", "LST", "LJF", "NONE")


def per_action_feature_names() -> list[str]:
    names = [
        "pa_token_budget_norm",
        "pa_n_admitted_prefill",
        "pa_n_decode_alloc",
        "pa_total_prefill_tokens",
        "pa_total_decode_tokens",
        "pa_strict_noop",          # 1 if no allocations and no eviction
        "pa_admits_anything",      # 1 if any prefill or decode allocated
        "pa_has_eviction",
        "pa_mapping_evict_idx",    # mapping[0] (action grid eviction index)
        "pa_mapping_budget_idx",   # mapping[1]
        "pa_mapping_heuristic_idx",# mapping[2]
    ]
    for r in EVICT_RULES:
        names.append(f"pa_evict__{r}")
    for h in ORDERING_HEURISTICS:
        names.append(f"pa_heur__{h}")
    return names


def _parse_action_repr(action_repr: str) -> dict:
    """Parse a ControllerAction repr string into a dict of fields.

    The repr is deterministic dataclass output so we can pluck fields by
    substring search. We only need the few we care about.
    """
    out: dict = {
        "token_budget": 0,
        "prefill_allocations_count": 0,
        "decode_allocations_count": 0,
        "prefill_total": 0,
        "decode_total": 0,
        "heuristic": None,
        "strategy": None,
        "mapping": (0, 0, 0),
    }
    if not action_repr:
        return out

    # token_budget=<N>
    s = action_repr
    try:
        i = s.index("token_budget=")
        end = s.index(",", i)
        out["token_budget"] = int(s[i + len("token_budget="): end])
    except ValueError:
        pass

    # prefill_allocations={...}
    try:
        i = s.index("prefill_allocations=")
        e = s.index("}", i) + 1
        body = s[i + len("prefill_allocations="): e]
        out["prefill_allocations_count"] = body.count(":")
        # Sum allocations: parse "k: v"
        if out["prefill_allocations_count"] > 0:
            kv = body.strip("{}").split(",")
            tot = 0
            for pair in kv:
                if ":" in pair:
                    try:
                        tot += int(pair.split(":", 1)[1].strip())
                    except Exception:
                        pass
            out["prefill_total"] = tot
    except ValueError:
        pass

    # decode_allocations={...}
    try:
        i = s.index("decode_allocations=")
        e = s.index("}", i) + 1
        body = s[i + len("decode_allocations="): e]
        out["decode_allocations_count"] = body.count(":")
        if out["decode_allocations_count"] > 0:
            kv = body.strip("{}").split(",")
            tot = 0
            for pair in kv:
                if ":" in pair:
                    try:
                        tot += int(pair.split(":", 1)[1].strip())
                    except Exception:
                        pass
            out["decode_total"] = tot
    except ValueError:
        pass

    # heuristic='<name>' or heuristic=None
    try:
        i = s.index("heuristic=")
        rest = s[i + len("heuristic="):]
        if rest.startswith("'"):
            j = rest.index("'", 1)
            out["heuristic"] = rest[1:j]
        elif rest.startswith("None"):
            out["heuristic"] = None
    except ValueError:
        pass

    # strategy='GV2|<rule>'
    try:
        i = s.index("strategy=")
        rest = s[i + len("strategy="):]
        if rest.startswith("'"):
            j = rest.index("'", 1)
            out["strategy"] = rest[1:j]
        elif rest.startswith("None"):
            out["strategy"] = None
    except ValueError:
        pass

    # mapping=(a, b, c)
    try:
        i = s.index("mapping=")
        rest = s[i + len("mapping="):]
        if rest.startswith("("):
            j = rest.index(")", 1)
            tup = rest[1:j].split(",")
            out["mapping"] = tuple(int(x.strip()) for x in tup if x.strip())
    except (ValueError, Exception):
        pass

    return out


def extract_per_action_features(action_repr: str) -> list[float]:
    """Return a fixed-length per-(s, a) feature vector from action_repr."""
    parsed = _parse_action_repr(action_repr)

    token_budget = int(parsed["token_budget"])
    n_admit_pre = int(parsed["prefill_allocations_count"])
    n_dec = int(parsed["decode_allocations_count"])
    pre_total = int(parsed["prefill_total"])
    dec_total = int(parsed["decode_total"])
    heur = parsed["heuristic"]
    strat = parsed["strategy"]
    mapping = parsed["mapping"]
    if len(mapping) < 3:
        mapping = tuple(list(mapping) + [0] * (3 - len(mapping)))

    # Eviction rule from strategy (e.g., "GV2|evict_none")
    rule = "evict_none"
    if isinstance(strat, str) and "|" in strat:
        rule = strat.split("|", 1)[1]

    strict_noop = 1.0 if (
        token_budget == 0 and n_admit_pre == 0 and n_dec == 0 and rule == "evict_none"
    ) else 0.0
    admits = 1.0 if (n_admit_pre > 0 or n_dec > 0) else 0.0
    has_evict = 1.0 if (rule != "evict_none") else 0.0

    feats = [
        float(token_budget) / 4096.0,
        float(n_admit_pre),
        float(n_dec),
        float(pre_total) / 4096.0,
        float(dec_total) / 4096.0,
        strict_noop,
        admits,
        has_evict,
        float(mapping[0]),
        float(mapping[1]),
        float(mapping[2]),
    ]
    # Eviction rule onehot
    for r in EVICT_RULES:
        feats.append(1.0 if r == rule else 0.0)
    # Heuristic onehot
    if heur is None:
        heur_label = "NONE"
    else:
        heur_label = str(heur).upper()
    for h in ORDERING_HEURISTICS:
        feats.append(1.0 if h == heur_label else 0.0)

    assert len(feats) == len(per_action_feature_names()), (
        f"per_action feature length mismatch: {len(feats)} vs "
        f"{len(per_action_feature_names())}"
    )
    return feats

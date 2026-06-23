from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List


@dataclass
class PrefillCreditLot:
    created_at: float
    expires_at: float
    tokens_remaining: int


class PrefillCreditStore:
    """
    Expiring credit store for prefill tokens.

    Policy:
    - Minted credits expire exactly `expiry_sec` after mint time.
    - Consumption is earliest-expiry-first (FIFO by mint order).
    """

    def __init__(self, *, expiry_sec: float, eps: float = 1e-9) -> None:
        if expiry_sec <= 0.0:
            raise ValueError("expiry_sec must be > 0")
        if eps <= 0.0:
            raise ValueError("eps must be > 0")
        self._expiry_sec = float(expiry_sec)
        self._eps = float(eps)
        self._lots: Deque[PrefillCreditLot] = deque()
        self._cached_total = 0  # sum(tokens_remaining) over non-expired lots

    @property
    def expiry_sec(self) -> float:
        return self._expiry_sec

    def clear(self) -> None:
        self._lots.clear()
        self._cached_total = 0

    def prune_expired(self, now: float) -> int:
        """
        Removes all expired lots.
        Returns number of expired tokens removed.
        """
        now_f = float(now)
        removed = 0
        while self._lots and (now_f + self._eps) >= self._lots[0].expires_at:
            lot = self._lots.popleft()
            if lot.tokens_remaining > 0:
                removed += int(lot.tokens_remaining)
                self._cached_total -= int(lot.tokens_remaining)
        if self._cached_total < 0:
            self._cached_total = 0
        return removed

    def mint(self, *, now: float, tokens: int) -> None:
        t = int(tokens)
        if t <= 0:
            return
        created = float(now)
        lot = PrefillCreditLot(
            created_at=created,
            expires_at=created + self._expiry_sec,
            tokens_remaining=t,
        )
        self._lots.append(lot)
        self._cached_total += t

    def available(self, *, now: float, prune: bool = True) -> int:
        if prune:
            self.prune_expired(now)
        return int(self._cached_total)

    def can_afford(self, *, now: float, tokens: int, prune: bool = True) -> bool:
        need = int(tokens)
        if need <= 0:
            return True
        return self.available(now=now, prune=prune) >= need

    def consume(self, *, now: float, tokens: int, prune: bool = True) -> int:
        """
        Consumes up to `tokens` credits.
        Returns actual consumed credits.
        """
        need = int(tokens)
        if need <= 0:
            return 0

        if prune:
            self.prune_expired(now)

        consumed = 0
        while need > 0 and self._lots:
            lot = self._lots[0]
            if lot.tokens_remaining <= 0:
                self._lots.popleft()
                continue

            take = min(need, lot.tokens_remaining)
            lot.tokens_remaining -= take
            consumed += take
            need -= take
            self._cached_total -= take

            if lot.tokens_remaining == 0:
                self._lots.popleft()

        if self._cached_total < 0:
            self._cached_total = 0
        return consumed

    def snapshot(self) -> Dict[str, object]:
        return {
            "expiry_sec": float(self._expiry_sec),
            "eps": float(self._eps),
            "lots": [
                {
                    "created_at": float(l.created_at),
                    "expires_at": float(l.expires_at),
                    "tokens_remaining": int(l.tokens_remaining),
                }
                for l in self._lots
                if l.tokens_remaining > 0
            ],
            "cached_total": int(self._cached_total),
        }

    @classmethod
    def from_snapshot(cls, payload: Dict[str, object]) -> "PrefillCreditStore":
        expiry_sec = float(payload["expiry_sec"])
        eps = float(payload.get("eps", 1e-9))
        out = cls(expiry_sec=expiry_sec, eps=eps)
        lots_raw = payload.get("lots", [])
        if isinstance(lots_raw, list):
            for item in lots_raw:
                if not isinstance(item, dict):
                    continue
                tr = int(item.get("tokens_remaining", 0))
                if tr <= 0:
                    continue
                out._lots.append(
                    PrefillCreditLot(
                        created_at=float(item.get("created_at", 0.0)),
                        expires_at=float(item.get("expires_at", 0.0)),
                        tokens_remaining=tr,
                    )
                )
                out._cached_total += tr
        # Recompute defensively (ignores malformed cached_total).
        out._cached_total = sum(max(0, int(l.tokens_remaining)) for l in out._lots)
        return out

    def debug_lots(self, *, now: float, prune: bool = True) -> List[Dict[str, float]]:
        if prune:
            self.prune_expired(now)
        out: List[Dict[str, float]] = []
        for l in self._lots:
            out.append(
                {
                    "created_at": float(l.created_at),
                    "expires_at": float(l.expires_at),
                    "tokens_remaining": float(l.tokens_remaining),
                }
            )
        return out


class DecodeCreditStore:
    """
    Non-expiring decode credit store.

    Intended semantics:
    - mint(216) when a request transitions from prefill -> decode
    - consume(1) for each decode token actually executed
    - balance must not go negative when `enforce_nonnegative=True`
    """

    def __init__(self, *, enforce_nonnegative: bool = True) -> None:
        self._balance = 0
        self._enforce_nonnegative = bool(enforce_nonnegative)

    @property
    def balance(self) -> int:
        return int(self._balance)

    @property
    def enforce_nonnegative(self) -> bool:
        return self._enforce_nonnegative

    def clear(self) -> None:
        self._balance = 0

    def mint(self, tokens: int) -> None:
        t = int(tokens)
        if t <= 0:
            return
        self._balance += t

    def can_afford(self, tokens: int) -> bool:
        need = int(tokens)
        if need <= 0:
            return True
        return self._balance >= need

    def consume(self, tokens: int) -> int:
        need = int(tokens)
        if need <= 0:
            return 0

        if self._enforce_nonnegative:
            take = min(need, self._balance)
            self._balance -= take
            return take

        # If nonnegative guard is disabled, allow temporary negative.
        self._balance -= need
        return need

    def snapshot(self) -> Dict[str, object]:
        return {
            "balance": int(self._balance),
            "enforce_nonnegative": bool(self._enforce_nonnegative),
        }

    @classmethod
    def from_snapshot(cls, payload: Dict[str, object]) -> "DecodeCreditStore":
        out = cls(enforce_nonnegative=bool(payload.get("enforce_nonnegative", True)))
        out._balance = int(payload.get("balance", 0))
        if out._enforce_nonnegative and out._balance < 0:
            out._balance = 0
        return out


@dataclass
class CreditStores:
    prefill: PrefillCreditStore
    decode: DecodeCreditStore

    def snapshot(self) -> Dict[str, object]:
        return {
            "prefill": self.prefill.snapshot(),
            "decode": self.decode.snapshot(),
        }

    @classmethod
    def from_snapshot(cls, payload: Dict[str, object]) -> "CreditStores":
        prefill_raw = payload.get("prefill", {})
        decode_raw = payload.get("decode", {})
        if not isinstance(prefill_raw, dict):
            prefill_raw = {}
        if not isinstance(decode_raw, dict):
            decode_raw = {}
        return cls(
            prefill=PrefillCreditStore.from_snapshot(prefill_raw),
            decode=DecodeCreditStore.from_snapshot(decode_raw),
        )

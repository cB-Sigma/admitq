"""In-process budget backend."""

from __future__ import annotations

import time
from collections.abc import Callable

from .base import Reservation, TenantLimits, bucket_adjust, bucket_reserve

__all__ = ["MemoryBackend"]

_SWEEP_AT = 10_000


class MemoryBackend:
    """Token buckets in a dict.

    State lives in this process only.  With several worker processes or replicas, each
    one enforces the configured budget independently, so a tenant can use up to
    ``workers x budget`` in total.  Use :class:`admitq.backends.redis.RedisBackend` when
    budgets must hold across processes.

    Not thread-safe; use it from one event loop.
    """

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.monotonic
        # tenant -> [tokens_left, requests_left, last_update]
        self._buckets: dict[str, list[float]] = {}

    def _state(self, tenant: str, limits: TenantLimits, now: float) -> list[float]:
        state = self._buckets.get(tenant)
        if state is None:
            if len(self._buckets) >= _SWEEP_AT:
                self._sweep(now, limits)
            state = self._buckets[tenant] = [
                limits.token_capacity,
                limits.request_capacity,
                now,
            ]
        return state

    def _sweep(self, now: float, limits: TenantLimits) -> None:
        # Buckets idle for longer than a full refill are indistinguishable from new ones.
        horizon = 2.0 * limits.burst_s
        self._buckets = {t: s for t, s in self._buckets.items() if now - s[2] < horizon}

    def reserve_sync(
        self, tenant: str, tokens: int, requests: int, limits: TenantLimits
    ) -> Reservation:
        now = self._clock()
        state = self._state(tenant, limits, now)
        state[0], state[1], decision = bucket_reserve(
            state[0], state[1], max(0.0, now - state[2]), tokens, requests, limits
        )
        state[2] = now
        return decision

    def adjust_sync(self, tenant: str, tokens: int, requests: int, limits: TenantLimits) -> None:
        now = self._clock()
        state = self._state(tenant, limits, now)
        state[0], state[1] = bucket_adjust(
            state[0], state[1], max(0.0, now - state[2]), tokens, requests, limits
        )
        state[2] = now

    async def reserve(
        self, tenant: str, tokens: int, requests: int, limits: TenantLimits
    ) -> Reservation:
        return self.reserve_sync(tenant, tokens, requests, limits)

    async def adjust(self, tenant: str, tokens: int, requests: int, limits: TenantLimits) -> None:
        self.adjust_sync(tenant, tokens, requests, limits)

    async def aclose(self) -> None:
        self._buckets.clear()

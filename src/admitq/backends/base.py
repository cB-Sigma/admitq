"""The budget backend interface and the token-bucket arithmetic both backends share."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

__all__ = ["BudgetBackend", "Reservation", "SyncBudgetBackend", "TenantLimits"]


@dataclass(frozen=True, slots=True)
class TenantLimits:
    """Budgets of one tenant.  ``None`` means unlimited in that dimension.

    Each budget is a token bucket that refills continuously at ``limit / 60`` per second
    and holds at most ``limit * burst_s / 60`` (one minute's worth by default).
    """

    tpm: float | None = None
    rpm: float | None = None
    burst_s: float = 60.0

    @property
    def unlimited(self) -> bool:
        return self.tpm is None and self.rpm is None

    @property
    def token_rate(self) -> float:
        return (self.tpm or 0.0) / 60.0

    @property
    def token_capacity(self) -> float:
        return (self.tpm or 0.0) * self.burst_s / 60.0

    @property
    def request_rate(self) -> float:
        return (self.rpm or 0.0) / 60.0

    @property
    def request_capacity(self) -> float:
        # A bucket that cannot hold one whole request would never admit anything.
        return max(1.0, self.rpm * self.burst_s / 60.0) if self.rpm else 0.0


@dataclass(frozen=True, slots=True)
class Reservation:
    """Outcome of :meth:`BudgetBackend.reserve`."""

    allowed: bool
    #: Which budget refused, when ``allowed`` is false.
    reason: Literal["tenant_tokens", "tenant_requests"] | None = None
    #: Seconds until the refused budget has refilled enough for this request.
    retry_after: float = 0.0
    #: Budget left after this call (``None`` if that dimension is unlimited).
    remaining_tokens: float | None = None
    remaining_requests: float | None = None


@runtime_checkable
class BudgetBackend(Protocol):
    """Stores per-tenant budgets.

    ``reserve`` must be atomic per tenant: check both buckets and debit both, or neither.
    ``adjust`` applies a signed correction (positive charges more, negative refunds) and
    must never fail a request - it runs after the fact.
    """

    async def reserve(
        self, tenant: str, tokens: int, requests: int, limits: TenantLimits
    ) -> Reservation: ...

    async def adjust(
        self, tenant: str, tokens: int, requests: int, limits: TenantLimits
    ) -> None: ...

    async def aclose(self) -> None: ...


@runtime_checkable
class SyncBudgetBackend(Protocol):
    """Optional fast path for backends that never block (the in-memory one)."""

    def reserve_sync(
        self, tenant: str, tokens: int, requests: int, limits: TenantLimits
    ) -> Reservation: ...

    def adjust_sync(
        self, tenant: str, tokens: int, requests: int, limits: TenantLimits
    ) -> None: ...


def bucket_reserve(
    tokens_left: float,
    requests_left: float,
    elapsed: float,
    tokens: float,
    requests: float,
    limits: TenantLimits,
) -> tuple[float, float, Reservation]:
    """Refill, check and debit.  Returns the new bucket levels and the decision.

    A request larger than the whole bucket is admitted when the bucket is full and drives
    it negative; the debt is then paid off by the refill.  That way an oversized request is
    slowed down rather than refused forever.
    """
    tok_rate, tok_cap = limits.token_rate, limits.token_capacity
    req_rate, req_cap = limits.request_rate, limits.request_capacity
    if tok_rate > 0:
        tokens_left = min(tok_cap, tokens_left + elapsed * tok_rate)
    if req_rate > 0:
        requests_left = min(req_cap, requests_left + elapsed * req_rate)

    wait_tokens = 0.0
    if tok_rate > 0:
        need = min(float(tokens), tok_cap)
        if tokens_left < need:
            wait_tokens = (need - tokens_left) / tok_rate
    wait_requests = 0.0
    if req_rate > 0:
        need = min(float(requests), req_cap)
        if requests_left < need:
            wait_requests = (need - requests_left) / req_rate

    if wait_tokens > 0 or wait_requests > 0:
        reason: Literal["tenant_tokens", "tenant_requests"] = (
            "tenant_tokens" if wait_tokens >= wait_requests else "tenant_requests"
        )
        return (
            tokens_left,
            requests_left,
            Reservation(
                allowed=False,
                reason=reason,
                retry_after=max(wait_tokens, wait_requests),
                remaining_tokens=max(0.0, tokens_left) if tok_rate > 0 else None,
                remaining_requests=max(0.0, requests_left) if req_rate > 0 else None,
            ),
        )
    if tok_rate > 0:
        tokens_left -= tokens
    if req_rate > 0:
        requests_left -= requests
    return (
        tokens_left,
        requests_left,
        Reservation(
            allowed=True,
            remaining_tokens=max(0.0, tokens_left) if tok_rate > 0 else None,
            remaining_requests=max(0.0, requests_left) if req_rate > 0 else None,
        ),
    )


def bucket_adjust(
    tokens_left: float,
    requests_left: float,
    elapsed: float,
    tokens: float,
    requests: float,
    limits: TenantLimits,
) -> tuple[float, float]:
    """Refill and apply a signed correction.  Debt is capped at one full bucket."""
    tok_rate, tok_cap = limits.token_rate, limits.token_capacity
    req_rate, req_cap = limits.request_rate, limits.request_capacity
    if tok_rate > 0:
        tokens_left = min(tok_cap, tokens_left + elapsed * tok_rate)
        tokens_left = max(-tok_cap, min(tok_cap, tokens_left - tokens))
    if req_rate > 0:
        requests_left = min(req_cap, requests_left + elapsed * req_rate)
        requests_left = max(-req_cap, min(req_cap, requests_left - requests))
    return tokens_left, requests_left

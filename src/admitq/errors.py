"""Shedding decisions and their HTTP representation."""

from __future__ import annotations

import math
from typing import Any, Literal

ShedReason = Literal[
    "tenant_tokens",
    "tenant_requests",
    "queue_full",
    "queue_timeout",
    "predicted_wait",
    "too_large",
    "backend_unavailable",
]

#: Reasons that are the caller's own budget (HTTP 429) rather than server overload (HTTP 503).
BUDGET_REASONS: frozenset[str] = frozenset({"tenant_tokens", "tenant_requests"})

_MESSAGES: dict[str, str] = {
    "tenant_tokens": "Token-per-minute budget exceeded for this tenant.",
    "tenant_requests": "Request-per-minute budget exceeded for this tenant.",
    "queue_full": "The server is overloaded: the admission queue is full.",
    "queue_timeout": "The server is overloaded: the request waited too long for capacity.",
    "predicted_wait": (
        "The server is overloaded: the estimated wait exceeds the queue deadline for this request."
    ),
    "too_large": "The request is larger than the configured in-flight token capacity.",
    "backend_unavailable": "The admission state backend is unavailable.",
}


class AdmitqError(Exception):
    """Base class for all errors raised by admitq."""


class ConfigError(AdmitqError, ValueError):
    """Raised for invalid configuration values."""


class Shed(AdmitqError):
    """A request was refused admission.

    Raised by :meth:`admitq.AdmissionController.acquire`; the ASGI middleware converts it
    into an HTTP response.  A shed request has never reached the downstream application.

    Attributes:
        reason: machine-readable reason, one of :data:`ShedReason`.
        status: HTTP status to use (429 for tenant budgets, 503 for overload by default).
        retry_after: seconds after which a retry has a reasonable chance.  Exact for
            tenant budgets (time until the bucket has refilled enough); an estimate derived
            from the measured drain rate for overload.
        tenant: the tenant the request was attributed to.
    """

    def __init__(
        self,
        reason: ShedReason,
        *,
        status: int,
        retry_after: float,
        tenant: str = "default",
        message: str | None = None,
    ) -> None:
        self.reason: ShedReason = reason
        self.status = status
        self.retry_after = max(0.0, retry_after)
        self.tenant = tenant
        self.message = message or _MESSAGES.get(reason, "Request refused by admission control.")
        super().__init__(f"{self.message} (reason={reason}, retry_after={self.retry_after:.2f}s)")

    @property
    def retry_after_header(self) -> str:
        """``Retry-After`` value: whole seconds, rounded up, at least 1."""
        return str(max(1, math.ceil(self.retry_after)))

    def body(self) -> dict[str, Any]:
        """OpenAI-style error body."""
        budget = self.reason in BUDGET_REASONS
        if self.reason == "too_large":
            err_type, code = "invalid_request_error", "request_too_large"
        elif budget:
            err_type, code = "rate_limit_error", "rate_limit_exceeded"
        else:
            err_type, code = "server_error", "server_overloaded"
        return {
            "error": {
                "message": self.message,
                "type": err_type,
                "param": None,
                "code": code,
                "reason": self.reason,
                "retry_after": round(self.retry_after, 3),
            }
        }

    def headers(self) -> dict[str, str]:
        """Headers to attach to the shed response."""
        return {
            "retry-after": self.retry_after_header,
            "x-admitq-shed-reason": self.reason,
        }

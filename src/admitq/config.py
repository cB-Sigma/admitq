"""Configuration for the admission controller."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from .backends.base import TenantLimits
from .errors import ConfigError

if TYPE_CHECKING:
    from .adaptive import AdaptiveLimit
    from .backends.base import BudgetBackend
    from .estimators import CostEstimator
    from .metrics import Recorder

__all__ = ["AdmissionConfig", "PriorityClass"]

DEFAULT_TENANT = "default"


@dataclass(frozen=True, slots=True)
class PriorityClass:
    """A priority class.

    Args:
        name: what requests call it (for example in the ``x-priority`` header).
        level: 0 is served first; larger numbers wait behind smaller ones.
        max_queue_s: queue deadline for this class; ``None`` uses the global ``max_queue_s``.
            Batch work usually tolerates a much longer wait than interactive work.
        aging_s: starvation protection.  A waiting request is treated as one level more
            urgent for every ``aging_s`` seconds it has waited.  ``None`` means half of the
            class's queue deadline, so every request spends the second half of its queue
            life competing one level up.  ``math.inf`` disables ageing (strict priority).
    """

    name: str
    level: int
    max_queue_s: float | None = None
    aging_s: float | None = None

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ConfigError("priority level must be >= 0")
        if self.max_queue_s is not None and self.max_queue_s < 0:
            raise ConfigError("max_queue_s must be >= 0")
        if self.aging_s is not None and not self.aging_s > 0:
            raise ConfigError("aging_s must be > 0 (use math.inf to disable ageing)")


_DEFAULT_PRIORITIES = (PriorityClass("interactive", 0), PriorityClass("batch", 1))


def _per_tenant(value: Mapping[str, float] | float | None, what: str) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        result = {str(k): float(v) for k, v in value.items()}
    else:
        result = {DEFAULT_TENANT: float(value)}
    for tenant, limit in result.items():
        if not limit > 0 or math.isinf(limit):
            raise ConfigError(f"{what}[{tenant!r}] must be a positive finite number")
    return result


@dataclass(kw_only=True)
class AdmissionConfig:
    """Everything :class:`admitq.AdmissionController` can be told.

    Capacity
        ``max_inflight_tokens`` / ``max_inflight_requests``: how much estimated work may
        be in flight at once.  Either, both or neither may be set.

    Queue
        ``max_queue_s``: longest a request may wait for capacity.  ``max_queue_depth``:
        most requests that may wait at once (``None`` = bounded by time only).
        ``early_shed``: refuse on arrival when the predicted wait already exceeds the
        request's deadline.

    Tenants
        ``tenant_tpm`` / ``tenant_rpm``: token-per-minute / request-per-minute budgets,
        as ``{tenant: limit}``.  The ``"default"`` entry applies to every tenant not
        listed (each gets its own bucket).  A bare number is shorthand for
        ``{"default": number}``.  ``tenant_weights``: fair-queuing weights (default 1).
        ``burst_s``: bucket size, in seconds of refill.  ``max_tenant_share``: hard cap on
        the fraction of in-flight tokens one tenant may hold.

    Priorities
        ``priorities``: ``{name: level}`` or a sequence of :class:`PriorityClass`.
        ``default_priority``: class for requests that name none (or an unknown one).

    Responses to shedding
        ``overload_status``: HTTP status for overload sheds (503 or 429).
        ``retry_after_min`` / ``retry_after_max``: clamp for ``Retry-After`` on overload.
    """

    max_inflight_tokens: int | None = None
    max_inflight_requests: int | None = None
    max_queue_s: float = 2.0
    max_queue_depth: int | None = 1024
    early_shed: bool = True
    oversize: Literal["clamp", "reject"] = "clamp"

    tenant_tpm: Mapping[str, float] | float | None = None
    tenant_rpm: Mapping[str, float] | float | None = None
    tenant_weights: Mapping[str, float] = field(default_factory=dict)
    burst_s: float = 60.0
    max_tenant_share: float | None = None

    priorities: Mapping[str, int] | Sequence[PriorityClass] = _DEFAULT_PRIORITIES
    default_priority: str | None = None

    overload_status: int = 503
    retry_after_min: float = 1.0
    retry_after_max: float = 60.0

    estimator: CostEstimator | None = None
    backend: BudgetBackend | None = None
    on_backend_error: Literal["open", "closed"] = "open"
    adaptive: AdaptiveLimit | None = None
    recorders: Sequence[Recorder] = ()
    max_tenant_labels: int = 100
    clock: Callable[[], float] | None = None

    def __post_init__(self) -> None:
        if self.max_inflight_tokens is not None and self.max_inflight_tokens < 1:
            raise ConfigError("max_inflight_tokens must be >= 1")
        if self.max_inflight_requests is not None and self.max_inflight_requests < 1:
            raise ConfigError("max_inflight_requests must be >= 1")
        if self.max_queue_s < 0 or math.isnan(self.max_queue_s):
            raise ConfigError("max_queue_s must be >= 0")
        if self.max_queue_depth is not None and self.max_queue_depth < 0:
            raise ConfigError("max_queue_depth must be >= 0")
        if self.oversize not in ("clamp", "reject"):
            raise ConfigError("oversize must be 'clamp' or 'reject'")
        if self.on_backend_error not in ("open", "closed"):
            raise ConfigError("on_backend_error must be 'open' or 'closed'")
        if self.overload_status not in (429, 503):
            raise ConfigError("overload_status must be 429 or 503")
        if not self.burst_s > 0:
            raise ConfigError("burst_s must be > 0")
        if self.max_tenant_share is not None and not 0.0 < self.max_tenant_share <= 1.0:
            raise ConfigError("max_tenant_share must be in (0, 1]")
        if self.retry_after_min < 0 or self.retry_after_max < self.retry_after_min:
            raise ConfigError("need 0 <= retry_after_min <= retry_after_max")
        if self.max_tenant_labels < 1:
            raise ConfigError("max_tenant_labels must be >= 1")

        self._tpm = _per_tenant(self.tenant_tpm, "tenant_tpm")
        self._rpm = _per_tenant(self.tenant_rpm, "tenant_rpm")
        self._weights = {str(k): float(v) for k, v in self.tenant_weights.items()}
        for tenant, weight in self._weights.items():
            if not weight > 0 or math.isinf(weight):
                raise ConfigError(f"tenant_weights[{tenant!r}] must be a positive finite number")

        if isinstance(self.priorities, Mapping):
            classes = [
                PriorityClass(str(name), int(level)) for name, level in self.priorities.items()
            ]
        else:
            classes = list(self.priorities)
        if not classes:
            raise ConfigError("at least one priority class is required")
        self._classes: dict[str, PriorityClass] = {}
        for cls in classes:
            if cls.name in self._classes:
                raise ConfigError(f"duplicate priority class {cls.name!r}")
            self._classes[cls.name] = cls
        if self.default_priority is None:
            self.default_priority = min(classes, key=lambda c: c.level).name
        elif self.default_priority not in self._classes:
            raise ConfigError(f"default_priority {self.default_priority!r} is not a priority class")
        self._limits_cache: dict[str, TenantLimits] = {}

    # -- lookups ----------------------------------------------------------------------

    def limits_for(self, tenant: str) -> TenantLimits:
        """Budgets that apply to ``tenant``."""
        cached = self._limits_cache.get(tenant)
        if cached is None:
            cached = TenantLimits(
                tpm=self._tpm.get(tenant, self._tpm.get(DEFAULT_TENANT)),
                rpm=self._rpm.get(tenant, self._rpm.get(DEFAULT_TENANT)),
                burst_s=self.burst_s,
            )
            if len(self._limits_cache) < 10_000:
                self._limits_cache[tenant] = cached
        return cached

    def weight_for(self, tenant: str) -> float:
        return self._weights.get(tenant, self._weights.get(DEFAULT_TENANT, 1.0))

    def priority_class(self, name: str | None) -> PriorityClass:
        """The class called ``name``, or the default class."""
        if name is not None:
            cls = self._classes.get(name)
            if cls is not None:
                return cls
        assert self.default_priority is not None
        return self._classes[self.default_priority]

    def queue_budget(self, cls: PriorityClass) -> float:
        return self.max_queue_s if cls.max_queue_s is None else cls.max_queue_s

    def aging_for(self, cls: PriorityClass) -> float:
        if cls.aging_s is not None:
            return cls.aging_s
        budget = self.queue_budget(cls)
        return budget / 2.0 if budget > 0 else math.inf

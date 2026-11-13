"""The admission controller: the programmatic API everything else is built on.

::

    controller = AdmissionController(max_inflight_tokens=64_000, max_queue_s=2.0)

    async with controller.admit(tenant="acme", cost=controller.estimate(body)) as ticket:
        response = await client.post(url, json=body)
        ticket.report_usage(**response.json()["usage"])

``admit`` raises :class:`admitq.Shed` when the request is refused.  Capacity is returned
when the ``async with`` block exits, however it exits.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping
from types import TracebackType
from typing import Any

from .backends.base import BudgetBackend, Reservation, SyncBudgetBackend, TenantLimits
from .backends.memory import MemoryBackend
from .config import AdmissionConfig, PriorityClass
from .errors import BUDGET_REASONS, Shed
from .estimators import CharEstimator, Cost, CostEstimator
from .metrics import Recorder, Registry
from .scheduler import Effects, Entry, EntryState, Scheduler, SchedulerSnapshot

__all__ = ["AdmissionController", "Ticket"]

logger = logging.getLogger("admitq")

_OTHER_TENANT = "_other"


class _Request:
    """Book-keeping for one submitted request (the scheduler entry's payload)."""

    __slots__ = (
        "cost",
        "entry",
        "future",
        "label",
        "limits",
        "priority",
        "reservation",
        "reserved",
        "tenant",
        "was_queued",
    )

    def __init__(
        self, tenant: str, label: str, priority: PriorityClass, cost: Cost, limits: TenantLimits
    ) -> None:
        self.tenant = tenant
        self.label = label
        self.priority = priority
        self.cost = cost
        self.limits = limits
        self.reservation: Reservation | None = None
        self.reserved = False
        self.entry: Entry | None = None
        self.future: asyncio.Future[None] | None = None
        self.was_queued = False


class Ticket:
    """An admitted request.  Holds capacity until :meth:`release` is called.

    Use it as an async context manager, or call :meth:`release` yourself in a ``finally``
    block.  :meth:`release` is idempotent, so doing both is harmless.
    """

    __slots__ = (
        "_actual",
        "_controller",
        "_first_byte",
        "_request",
        "_source",
        "outcome",
        "released",
    )

    def __init__(self, controller: AdmissionController, request: _Request) -> None:
        self._controller = controller
        self._request = request
        self._actual: tuple[int | None, int | None] | None = None
        self._source = "estimate"
        self._first_byte = False
        self.released = False
        #: Outcome passed to :meth:`release` (``None`` until released).
        self.outcome: str | None = None

    # -- read-only facts --------------------------------------------------------------

    @property
    def tenant(self) -> str:
        return self._request.tenant

    @property
    def priority(self) -> str:
        return self._request.priority.name

    @property
    def cost(self) -> Cost:
        """The pre-admission estimate."""
        return self._request.cost

    @property
    def queue_time(self) -> float:
        """Seconds this request waited for capacity."""
        entry = self._request.entry
        return entry.queue_time if entry is not None else 0.0

    @property
    def remaining_tokens(self) -> float | None:
        """Tenant token budget left after this request's reservation (``None`` = unlimited
        or unknown)."""
        reservation = self._request.reservation
        return reservation.remaining_tokens if reservation is not None else None

    @property
    def remaining_requests(self) -> float | None:
        reservation = self._request.reservation
        return reservation.remaining_requests if reservation is not None else None

    @property
    def limits(self) -> TenantLimits:
        return self._request.limits

    @property
    def actual_tokens(self) -> int | None:
        """Total tokens charged at release (``None`` until released)."""
        if not self.released:
            return None
        return self._resolve_actual(self.outcome or "ok")

    # -- reporting --------------------------------------------------------------------

    def report_usage(
        self,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        source: str = "usage",
        **_ignored: Any,
    ) -> None:
        """Record what the request really cost.

        Accepts the fields of an OpenAI ``usage`` object directly
        (``ticket.report_usage(**response["usage"])``); unknown keys are ignored.  Whatever
        is not given falls back to the estimate.  The correction is applied at release.
        """
        if completion_tokens is None and total_tokens is not None and prompt_tokens is not None:
            completion_tokens = max(0, total_tokens - prompt_tokens)
        if prompt_tokens is None and completion_tokens is None:
            return
        self._actual = (prompt_tokens, completion_tokens)
        self._source = source

    def first_byte(self) -> None:
        """Mark the arrival of the first response byte (feeds the time-to-first-byte
        histogram and the adaptive limit).  Only the first call counts."""
        if self._first_byte or self.released:
            return
        self._first_byte = True
        self._controller._on_first_byte(self._request)

    def release(self, outcome: str = "ok") -> bool:
        """Return capacity and settle the budget.  Returns ``False`` if already released.

        ``outcome`` is a label for metrics: ``"ok"``, ``"error"``, ``"client_disconnect"``,
        ``"cancelled"``.  If no usage was reported, a successful request is charged its
        estimate and an unsuccessful one is charged nothing.
        """
        if self.released:
            return False
        self.released = True
        self.outcome = outcome
        self._controller._release(
            self._request, self._resolve_actual(outcome), self._source, outcome
        )
        return True

    def _resolve_actual(self, outcome: str) -> int:
        cost = self._request.cost
        if self._actual is not None:
            prompt, completion = self._actual
            return (cost.prompt_tokens if prompt is None else prompt) + (
                cost.output_tokens if completion is None else completion
            )
        return cost.total if outcome == "ok" else 0

    async def __aenter__(self) -> Ticket:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.release("ok")
        elif issubclass(exc_type, asyncio.CancelledError):
            self.release("cancelled")
        else:
            self.release("error")


class _Admission:
    """Returned by :meth:`AdmissionController.admit`; awaitable and an async context manager."""

    __slots__ = ("_controller", "_kwargs", "_ticket")

    def __init__(self, controller: AdmissionController, kwargs: dict[str, Any]) -> None:
        self._controller = controller
        self._kwargs = kwargs
        self._ticket: Ticket | None = None

    async def __aenter__(self) -> Ticket:
        self._ticket = await self._controller.acquire(**self._kwargs)
        return self._ticket

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._ticket is not None:
            await self._ticket.__aexit__(exc_type, exc, tb)


class AdmissionController:
    """Token-aware admission control with tenant budgets, fair queuing and priorities.

    Construct it with an :class:`AdmissionConfig` or with the same fields as keyword
    arguments.  One controller belongs to one event loop.
    """

    def __init__(self, config: AdmissionConfig | None = None, **options: Any) -> None:
        if config is not None and options:
            raise TypeError("pass either a config object or keyword options, not both")
        self.config = config if config is not None else AdmissionConfig(**options)
        cfg = self.config
        self._clock = cfg.clock
        self._loop: asyncio.AbstractEventLoop | None = None
        self.estimator: CostEstimator = cfg.estimator or CharEstimator()
        self.backend: BudgetBackend = cfg.backend or MemoryBackend(clock=self._now)
        self._sync_backend: SyncBudgetBackend | None = (
            self.backend if isinstance(self.backend, SyncBudgetBackend) else None
        )
        #: Built-in metric registry; ``controller.metrics.render()`` is Prometheus text.
        self.metrics = Registry()
        self._recorders: tuple[Recorder, ...] = (self.metrics, *cfg.recorders)
        self._labels: set[str] = set()

        limit_tokens = cfg.max_inflight_tokens
        limit_requests = cfg.max_inflight_requests
        self.adaptive = cfg.adaptive
        self._adaptive_dimension = "tokens"
        if self.adaptive is not None:
            dimension = self.adaptive.dimension or (
                "tokens" if limit_tokens is not None or limit_requests is None else "requests"
            )
            self._adaptive_dimension = dimension
            start = self.adaptive.start(
                limit_tokens if dimension == "tokens" else limit_requests, self._now()
            )
            if dimension == "tokens":
                limit_tokens = start
            else:
                limit_requests = start

        self.scheduler = Scheduler(
            limit_tokens=limit_tokens,
            limit_requests=limit_requests,
            max_queue_depth=cfg.max_queue_depth,
            max_tenant_share=cfg.max_tenant_share,
            oversize=cfg.oversize,
            early_shed=cfg.early_shed,
            retry_after_min=cfg.retry_after_min,
            retry_after_max=cfg.retry_after_max,
        )
        self._timer: asyncio.TimerHandle | None = None
        self._timer_at: float | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._closed = False
        self._update_gauges()

    # ------------------------------------------------------------------ public API

    def estimate(self, body: Mapping[str, Any], endpoint: str = "chat") -> Cost:
        """Estimate the cost of a request body with the configured estimator."""
        return self.estimator.estimate(body, endpoint)

    def admit(
        self,
        *,
        tenant: str = "default",
        cost: Cost | int,
        priority: str | None = None,
        timeout: float | None = None,
    ) -> _Admission:
        """``async with controller.admit(...) as ticket:`` - acquire, run, release."""
        return _Admission(
            self, {"tenant": tenant, "cost": cost, "priority": priority, "timeout": timeout}
        )

    async def acquire(
        self,
        *,
        tenant: str = "default",
        cost: Cost | int,
        priority: str | None = None,
        timeout: float | None = None,
    ) -> Ticket:
        """Wait for admission and return a :class:`Ticket`, or raise :class:`Shed`.

        Args:
            tenant: who is asking; selects budgets and the fair-queuing account.
            cost: estimated tokens (a :class:`Cost` or a plain total).
            priority: name of a configured priority class; unknown names get the default.
            timeout: the caller's own patience in seconds.  The request is never queued
                for longer than ``min(max_queue_s, timeout)``.

        If the awaiting task is cancelled, the request is withdrawn and nothing is charged.
        """
        request = await self._submit(tenant, cost, priority, timeout)
        if request.future is not None:
            try:
                await request.future
            except asyncio.CancelledError:
                self._abandon(request)
                raise
        return Ticket(self, request)

    def report_signal(self, value: float) -> None:
        """Feed an engine congestion signal (for example a polled queue depth) to the
        adaptive limit.  Ignored unless ``adaptive`` is configured with
        ``signal="reported"``."""
        if self.adaptive is not None and self.adaptive.signal == "reported":
            self._adapt(value)

    def snapshot(self) -> SchedulerSnapshot:
        """Current capacity and queue state."""
        return self.scheduler.snapshot()

    async def aclose(self) -> None:
        """Shed everything still queued, wait for pending budget settlements and close
        the backend."""
        self._apply(self.scheduler.expire(float("inf")))
        self._closed = True
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.backend.aclose()

    # ------------------------------------------------------------------ internals

    def _now(self) -> float:
        if self._clock is not None:
            return self._clock()
        loop = self._loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return time.monotonic()
        return loop.time()

    def _label(self, tenant: str) -> str:
        """Tenant label for metrics, bounded in cardinality."""
        labels = self._labels
        if tenant in labels:
            return tenant
        if len(labels) < self.config.max_tenant_labels:
            labels.add(tenant)
            return tenant
        return _OTHER_TENANT

    def _inc(self, name: str, labels: Mapping[str, str], value: float = 1.0) -> None:
        for recorder in self._recorders:
            recorder.inc(name, labels, value)

    def _observe(self, name: str, labels: Mapping[str, str], value: float) -> None:
        for recorder in self._recorders:
            recorder.observe(name, labels, value)

    def _update_gauges(self) -> None:
        s = self.scheduler
        values = (
            ("admitq_inflight_tokens", s.inflight_tokens),
            ("admitq_inflight_requests", s.inflight_requests),
            ("admitq_queue_depth", s.queued_requests),
            ("admitq_queued_tokens", s.queued_tokens),
            ("admitq_inflight_token_limit", s.limit_tokens or 0),
            ("admitq_inflight_request_limit", s.limit_requests or 0),
        )
        for recorder in self._recorders:
            for name, value in values:
                recorder.set(name, {}, float(value))

    async def _submit(
        self, tenant: str, cost: Cost | int, priority: str | None, timeout: float | None
    ) -> _Request:
        """Reserve budget and offer the request to the scheduler.

        Returns a request that is either admitted (``future is None``) or queued
        (``future`` resolves on admission and raises :class:`Shed` on shedding).  Raises
        :class:`Shed` for requests refused on arrival.
        """
        if self._closed:
            raise RuntimeError("AdmissionController is closed")
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        cfg = self.config
        if not isinstance(cost, Cost):
            cost = Cost(prompt_tokens=int(cost))
        cls = cfg.priority_class(priority)
        limits = cfg.limits_for(tenant)
        request = _Request(tenant, self._label(tenant), cls, cost, limits)

        if not limits.unlimited:
            try:
                if self._sync_backend is not None:
                    reservation = self._sync_backend.reserve_sync(tenant, cost.total, 1, limits)
                else:
                    reservation = await self.backend.reserve(tenant, cost.total, 1, limits)
            except Exception:
                logger.warning("admitq: budget backend reserve failed", exc_info=True)
                self._inc("admitq_backend_errors_total", {"operation": "reserve"})
                if cfg.on_backend_error == "closed":
                    self._count_shed(request, "backend_unavailable")
                    raise Shed(
                        "backend_unavailable",
                        status=503,
                        retry_after=cfg.retry_after_min,
                        tenant=tenant,
                    ) from None
            else:
                request.reservation = reservation
                if not reservation.allowed:
                    reason = reservation.reason or "tenant_tokens"
                    self._count_shed(request, reason)
                    raise Shed(
                        reason, status=429, retry_after=reservation.retry_after, tenant=tenant
                    )
                request.reserved = True

        now = self._now()
        budget = cfg.queue_budget(cls)
        if timeout is not None:
            budget = min(budget, max(0.0, timeout))
        entry, effects = self.scheduler.offer(
            tenant=tenant,
            cost=cost.total,
            now=now,
            deadline=now + budget,
            level=cls.level,
            weight=cfg.weight_for(tenant),
            aging_s=cfg.aging_for(cls),
            payload=request,
        )
        request.entry = entry
        self._apply(effects)
        if entry.state is EntryState.SHED:
            raise self._shed_error(request)
        if entry.state is EntryState.QUEUED:
            request.was_queued = True
            request.future = self._loop.create_future()
            self._inc("admitq_queued_total", {"tenant": request.label, "priority": cls.name})
            self._arm_timer()
        return request

    def _shed_error(self, request: _Request) -> Shed:
        entry = request.entry
        assert entry is not None
        assert entry.shed_reason is not None
        reason = entry.shed_reason
        if reason in BUDGET_REASONS:
            status = 429
        elif reason == "too_large":
            status = 400
        else:
            status = self.config.overload_status
        return Shed(
            reason,
            status=status,
            retry_after=entry.retry_after if entry.retry_after is not None else 0.0,
            tenant=request.tenant,
        )

    def _count_shed(self, request: _Request, reason: str) -> None:
        self._inc(
            "admitq_shed_total",
            {"tenant": request.label, "priority": request.priority.name, "reason": reason},
        )

    def _apply(self, effects: Effects) -> None:
        """Deliver scheduler decisions to waiters, metrics and budgets."""
        if not effects:
            self._update_gauges()
            return
        for entry in effects.admitted:
            request: _Request = entry.payload
            priority = request.priority.name
            self._inc("admitq_admitted_total", {"tenant": request.label, "priority": priority})
            self._inc(
                "admitq_estimated_tokens_total", {"tenant": request.label}, request.cost.total
            )
            self._observe("admitq_queue_time_seconds", {"priority": priority}, entry.queue_time)
            future = request.future
            if future is not None and not future.done():
                future.set_result(None)
        for entry in effects.shed:
            request = entry.payload
            self._count_shed(request, entry.shed_reason or "queue_timeout")
            self._refund(request)
            future = request.future
            if future is not None and not future.done():
                future.set_exception(self._shed_error(request))
        if self.adaptive is not None and effects.admitted:
            s = self.scheduler
            tokens = self._adaptive_dimension == "tokens"
            for entry in effects.admitted:
                self.adaptive.note_inflight(
                    s.inflight_tokens if tokens else s.inflight_requests,
                    float(entry.charged) if tokens else 1.0,
                )
        self._update_gauges()
        self._arm_timer()

    def _abandon(self, request: _Request) -> None:
        """The waiter went away (cancelled or client disconnected) before using capacity."""
        entry = request.entry
        if entry is None:
            return
        future = request.future
        if future is not None and future.done() and not future.cancelled():
            # Consume a pending Shed so asyncio does not log "exception was never retrieved".
            future.exception()
        if entry.state is EntryState.QUEUED:
            self._apply(self.scheduler.cancel(entry, self._now()))
            self._refund(request)
            self._inc(
                "admitq_completed_total", {"tenant": request.label, "outcome": "abandoned_in_queue"}
            )
        elif entry.state is EntryState.ADMITTED:
            # Admitted in the same instant the waiter was cancelled: give it straight back.
            self._release(request, 0, "estimate", "cancelled")

    def _refund(self, request: _Request) -> None:
        if request.reserved:
            request.reserved = False
            self._adjust(request, -request.cost.total, -1)

    def _adjust(self, request: _Request, tokens: int, requests: int) -> None:
        if tokens == 0 and requests == 0:
            return
        if self._sync_backend is not None:
            self._sync_backend.adjust_sync(request.tenant, tokens, requests, request.limits)
            return
        if self._loop is None or self._closed:
            return
        task = self._loop.create_task(self._adjust_async(request, tokens, requests))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _adjust_async(self, request: _Request, tokens: int, requests: int) -> None:
        try:
            await self.backend.adjust(request.tenant, tokens, requests, request.limits)
        except Exception:
            logger.warning("admitq: budget backend adjust failed", exc_info=True)
            self._inc("admitq_backend_errors_total", {"operation": "adjust"})

    def _release(self, request: _Request, actual: int, source: str, outcome: str) -> None:
        entry = request.entry
        assert entry is not None
        if entry.state is not EntryState.ADMITTED:
            return
        estimated = request.cost.total
        effects = self.scheduler.release(entry, self._now(), actual_cost=actual)
        self._inc("admitq_completed_total", {"tenant": request.label, "outcome": outcome})
        self._inc("admitq_actual_tokens_total", {"tenant": request.label, "source": source}, actual)
        if source != "estimate" and estimated > 0:
            self._observe("admitq_estimate_ratio", {"source": source}, actual / estimated)
        if request.reserved:
            request.reserved = False
            self._adjust(request, actual - estimated, 0)
        self._apply(effects)

    def _on_first_byte(self, request: _Request) -> None:
        entry = request.entry
        if entry is None or entry.admitted_at is None:
            return
        ttfb = max(0.0, self._now() - entry.admitted_at)
        self._observe("admitq_time_to_first_byte_seconds", {}, ttfb)
        if self.adaptive is not None and self.adaptive.signal == "ttfb":
            self._adapt(ttfb)

    def _adapt(self, value: float) -> None:
        assert self.adaptive is not None
        now = self._now()
        new_limit = self.adaptive.record(value, now)
        if new_limit is None:
            return
        if self._adaptive_dimension == "tokens":
            effects = self.scheduler.set_limits(now, limit_tokens=new_limit)
        else:
            effects = self.scheduler.set_limits(now, limit_requests=new_limit)
        self._apply(effects)

    def _arm_timer(self) -> None:
        deadline = self.scheduler.next_deadline()
        if deadline is None or self._loop is None or self._closed:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
                self._timer_at = None
            return
        if self._timer is not None:
            if self._timer_at == deadline:
                return
            self._timer.cancel()
        self._timer_at = deadline
        self._timer = self._loop.call_later(max(0.0, deadline - self._now()), self._on_timer)

    def _on_timer(self) -> None:
        # The loop may fire a timer a clock tick early; never let that turn into a re-arm
        # spin - treat the armed deadline as reached.
        now = self._now()
        if self._timer_at is not None and now < self._timer_at:
            now = self._timer_at
        self._timer = None
        self._timer_at = None
        try:
            self._apply(self.scheduler.expire(now))
        except Exception:  # pragma: no cover - defensive: a timer must never die silently
            logger.exception("admitq: queue expiry failed")
        self._arm_timer()

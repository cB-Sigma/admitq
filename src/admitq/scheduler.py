"""The admission scheduler: a synchronous, clock-driven state machine.

It owns three things:

* the **capacity** - in-flight tokens and in-flight requests, each with an optional limit;
* the **queue** - entries waiting for capacity, each with a deadline;
* the **order** - which waiting entry gets capacity next (priority level with ageing, then
  weighted fair queuing between tenants, then arrival order).

Every method takes the current time as an argument and returns the side effects it caused
(:class:`Effects`), so the whole thing is deterministic and testable without an event loop.
:class:`admitq.AdmissionController` wraps it with futures, timers and budgets.

The scheduler is not thread-safe; drive it from one thread (one event loop).

Fair queuing
------------
Start-time fair queuing over tenants.  Every tenant has a virtual time ``vt`` that advances
by ``cost / weight`` whenever one of its requests is admitted.  The next request comes from
the eligible tenant with the smallest ``max(V, vt)``, where ``V`` is the global virtual
time.  A tenant that was idle re-enters at ``V``: it gets no credit for the past, and a
tenant that used capacity while nobody else wanted it carries no debt.

Head-of-line policy
-------------------
The chosen entry is admitted only if it fits.  Smaller entries behind it do *not* jump
ahead - that would let a stream of small requests starve a large one.  The price is some
idle capacity while a large request waits for room.
"""

from __future__ import annotations

import enum
import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from .errors import ShedReason

__all__ = ["Effects", "Entry", "EntryState", "Scheduler", "SchedulerSnapshot"]

_EWMA_ALPHA = 0.1
_PRUNE_THRESHOLD = 4096


class EntryState(enum.Enum):
    QUEUED = "queued"
    ADMITTED = "admitted"
    RELEASED = "released"
    SHED = "shed"
    CANCELLED = "cancelled"


@dataclass(slots=True, eq=False)
class Entry:
    """One request as the scheduler sees it."""

    id: int
    tenant: str
    level: int
    cost: int
    weight: float
    enqueued_at: float
    deadline: float
    aging_s: float
    state: EntryState = EntryState.QUEUED
    admitted_at: float | None = None
    #: Tokens this entry currently holds against the in-flight limit.
    charged: int = 0
    shed_reason: ShedReason | None = None
    #: Estimated seconds until a retry could be admitted; set when the entry is shed.
    retry_after: float | None = None
    #: Opaque slot for the caller (the controller stores its future here).
    payload: Any = None

    @property
    def queue_time(self) -> float:
        """Seconds spent queued; 0 until admitted."""
        if self.admitted_at is None:
            return 0.0
        return max(0.0, self.admitted_at - self.enqueued_at)


@dataclass(slots=True)
class Effects:
    """Entries whose state changed as a result of one scheduler call."""

    admitted: list[Entry] = field(default_factory=list)
    shed: list[Entry] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.admitted or self.shed)


@dataclass(frozen=True, slots=True)
class SchedulerSnapshot:
    inflight_tokens: int
    inflight_requests: int
    limit_tokens: int | None
    limit_requests: int | None
    queued_requests: int
    queued_tokens: int
    #: Token-weighted mean time a request holds capacity, or ``None`` before any release.
    mean_hold_s: float | None


_Flow = tuple[str, int]


class Scheduler:
    """Token- and request-bounded admission with a deadline-bounded fair queue.

    Args:
        limit_tokens: maximum tokens in flight, or ``None`` for no token limit.
        limit_requests: maximum requests in flight, or ``None`` for no request limit.
        max_queue_depth: maximum number of waiting entries, or ``None`` (the queue is then
            bounded only by time).
        max_tenant_share: if set, no tenant may hold more than this fraction of
            ``limit_tokens`` in flight (a tenant with nothing in flight may always run one
            request).  Not work-conserving: capacity can sit idle to keep room for others.
        oversize: what to do with a request whose cost exceeds ``limit_tokens``: ``"clamp"``
            charges it the whole limit (it runs alone), ``"reject"`` sheds it.
        early_shed: shed on arrival when the predicted wait already exceeds the entry's
            deadline, instead of letting it sit in the queue and time out.
    """

    def __init__(
        self,
        *,
        limit_tokens: int | None = None,
        limit_requests: int | None = None,
        max_queue_depth: int | None = None,
        max_tenant_share: float | None = None,
        oversize: Literal["clamp", "reject"] = "clamp",
        early_shed: bool = True,
        retry_after_min: float = 1.0,
        retry_after_max: float = 60.0,
    ) -> None:
        if limit_tokens is not None and limit_tokens < 1:
            raise ValueError("limit_tokens must be >= 1")
        if limit_requests is not None and limit_requests < 1:
            raise ValueError("limit_requests must be >= 1")
        if max_queue_depth is not None and max_queue_depth < 0:
            raise ValueError("max_queue_depth must be >= 0")
        if max_tenant_share is not None and not 0.0 < max_tenant_share <= 1.0:
            raise ValueError("max_tenant_share must be in (0, 1]")
        self.limit_tokens = limit_tokens
        self.limit_requests = limit_requests
        self.max_queue_depth = max_queue_depth
        self.max_tenant_share = max_tenant_share
        self.oversize = oversize
        self.early_shed = early_shed
        self.retry_after_min = retry_after_min
        self.retry_after_max = retry_after_max

        self.inflight_tokens = 0
        self.inflight_requests = 0
        self.queued_requests = 0
        self.queued_tokens = 0

        self._seq = 0
        self._flows: dict[_Flow, deque[Entry]] = {}
        self._flow_tokens: dict[_Flow, int] = {}
        self._deadlines: list[tuple[float, int, Entry]] = []
        self._vtime: dict[str, float] = {}
        self._v = 0.0
        self._tenant_inflight: dict[str, int] = {}
        self._tenant_requests: dict[str, int] = {}
        # Exponentially decayed means of how long an admitted request holds capacity:
        # token-weighted (for the token limit) and per request (for the request limit).
        self._hold_tok_num = 0.0
        self._hold_tok_den = 0.0
        self._hold_req_num = 0.0
        self._hold_req_den = 0.0

    # ------------------------------------------------------------------ public API

    def offer(
        self,
        *,
        tenant: str,
        cost: int,
        now: float,
        deadline: float,
        level: int = 0,
        weight: float = 1.0,
        aging_s: float = math.inf,
        payload: Any = None,
    ) -> tuple[Entry, Effects]:
        """Submit a request.

        Returns the new entry and the effects of submitting it.  Inspect ``entry.state``:
        ``ADMITTED`` (holds capacity now), ``QUEUED`` (waiting; it will later show up in
        the effects of another call) or ``SHED`` (see ``entry.shed_reason``).
        """
        if cost < 0:
            raise ValueError("cost must be non-negative")
        if weight <= 0:
            raise ValueError("weight must be positive")
        effects = Effects()
        self._expire(now, effects)

        self._seq += 1
        entry = Entry(
            id=self._seq,
            tenant=tenant,
            level=level,
            cost=cost,
            weight=weight,
            enqueued_at=now,
            deadline=deadline,
            aging_s=aging_s,
            payload=payload,
        )
        if self.limit_tokens is not None and cost > self.limit_tokens and self.oversize == "reject":
            self._mark_shed(entry, "too_large", now, effects, retry_after=0.0)
            return entry, effects

        self._enqueue(entry)
        self._dispatch(now, effects)
        if entry.state is not EntryState.QUEUED:
            return entry, effects

        # The entry has to wait.  Decide whether it is allowed to.
        if entry.deadline <= now:
            self._remove(entry)
            self._mark_shed(entry, "queue_timeout", now, effects)
            return entry, effects
        if self.max_queue_depth is not None and self.queued_requests > self.max_queue_depth:
            victim = self._eviction_victim(entry)
            self._remove(victim)
            self._mark_shed(victim, "queue_full", now, effects)
            if victim is entry:
                return entry, effects
        if self.early_shed:
            wait = self.predicted_wait(entry)
            if wait is not None and wait > entry.deadline - now:
                self._remove(entry)
                self._mark_shed(entry, "predicted_wait", now, effects, wait=wait)
                # Removing it may unblock nothing new, but keep the invariant simple.
                self._dispatch(now, effects)
        return entry, effects

    def release(self, entry: Entry, now: float, actual_cost: int | None = None) -> Effects:
        """Return an admitted entry's capacity.  Idempotent: later calls do nothing.

        ``actual_cost`` (if known) corrects the tenant's fair-queuing account so that
        fairness follows real usage rather than estimates.
        """
        effects = Effects()
        if entry.state is not EntryState.ADMITTED:
            return effects
        entry.state = EntryState.RELEASED
        self.inflight_tokens -= entry.charged
        self.inflight_requests -= 1
        count = self._tenant_requests.get(entry.tenant, 0) - 1
        if count > 0:
            self._tenant_requests[entry.tenant] = count
            self._tenant_inflight[entry.tenant] -= entry.charged
        else:
            self._tenant_requests.pop(entry.tenant, None)
            self._tenant_inflight.pop(entry.tenant, None)

        hold = max(0.0, now - (entry.admitted_at if entry.admitted_at is not None else now))
        decay = 1.0 - _EWMA_ALPHA
        self._hold_req_num = self._hold_req_num * decay + hold
        self._hold_req_den = self._hold_req_den * decay + 1.0
        if entry.charged > 0:
            self._hold_tok_num = self._hold_tok_num * decay + hold * entry.charged
            self._hold_tok_den = self._hold_tok_den * decay + entry.charged

        if actual_cost is not None and entry.tenant in self._vtime:
            self._vtime[entry.tenant] += (actual_cost - entry.cost) / entry.weight
        entry.charged = 0
        self._expire(now, effects)
        self._dispatch(now, effects)
        self._maybe_prune()
        return effects

    def cancel(self, entry: Entry, now: float) -> Effects:
        """Withdraw an entry: dequeue it if waiting, release it if admitted."""
        if entry.state is EntryState.ADMITTED:
            return self.release(entry, now)
        effects = Effects()
        if entry.state is EntryState.QUEUED:
            self._remove(entry)
            entry.state = EntryState.CANCELLED
            self._expire(now, effects)
            self._dispatch(now, effects)
        return effects

    def expire(self, now: float) -> Effects:
        """Shed every waiting entry whose deadline has passed, then admit what now fits."""
        effects = Effects()
        self._expire(now, effects)
        self._dispatch(now, effects)
        return effects

    def set_limits(
        self,
        now: float,
        *,
        limit_tokens: int | None = None,
        limit_requests: int | None = None,
    ) -> Effects:
        """Change limits at runtime (``None`` leaves a limit unchanged).

        Lowering a limit never evicts in-flight work; the scheduler simply admits nothing
        until enough has finished.
        """
        if limit_tokens is not None:
            if limit_tokens < 1:
                raise ValueError("limit_tokens must be >= 1")
            self.limit_tokens = limit_tokens
        if limit_requests is not None:
            if limit_requests < 1:
                raise ValueError("limit_requests must be >= 1")
            self.limit_requests = limit_requests
        return self.expire(now)

    def next_deadline(self) -> float | None:
        """Earliest deadline among waiting entries, or ``None`` if nothing is waiting."""
        heap = self._deadlines
        while heap and heap[0][2].state is not EntryState.QUEUED:
            heapq.heappop(heap)
        return heap[0][0] if heap else None

    def snapshot(self) -> SchedulerSnapshot:
        return SchedulerSnapshot(
            inflight_tokens=self.inflight_tokens,
            inflight_requests=self.inflight_requests,
            limit_tokens=self.limit_tokens,
            limit_requests=self.limit_requests,
            queued_requests=self.queued_requests,
            queued_tokens=self.queued_tokens,
            mean_hold_s=self._hold_tok if self._hold_tok is not None else self._hold_req,
        )

    @property
    def _hold_tok(self) -> float | None:
        return self._hold_tok_num / self._hold_tok_den if self._hold_tok_den > 0 else None

    @property
    def _hold_req(self) -> float | None:
        return self._hold_req_num / self._hold_req_den if self._hold_req_den > 0 else None

    def tenant_inflight_tokens(self, tenant: str) -> int:
        return self._tenant_inflight.get(tenant, 0)

    def predicted_wait(self, entry: Entry) -> float | None:
        """Estimated seconds until ``entry`` (queued) would be admitted.

        Little's law: with ``N`` tokens in flight held for ``T`` seconds on average, the
        system drains ``N / T`` tokens per second, so work ``W`` ahead of the entry takes
        about ``W * T / N``.  The work ahead respects the scheduling order: everything at a
        better priority level, plus what fair queuing lets other tenants take first at the
        same level.  Returns ``None`` until at least one request has completed (no rate to
        extrapolate from) or when no limit is configured.
        """
        if self.limit_tokens is None and self.limit_requests is None:
            return None
        hold_tok, hold_req = self._hold_tok, self._hold_req
        if hold_tok is None and hold_req is None:
            return None
        ahead_tokens, ahead_requests = self._work_ahead(entry)
        wait = 0.0
        if self.limit_tokens is not None and hold_tok is not None:
            free = max(0, self.limit_tokens - self.inflight_tokens)
            need = ahead_tokens + min(entry.cost, self.limit_tokens) - free
            if need > 0:
                wait = max(wait, need * hold_tok / self.limit_tokens)
        if self.limit_requests is not None and hold_req is not None:
            free_slots = max(0, self.limit_requests - self.inflight_requests)
            need_slots = ahead_requests + 1 - free_slots
            if need_slots > 0:
                wait = max(wait, need_slots * hold_req / self.limit_requests)
        return wait

    # ------------------------------------------------------------------ internals

    def _charge_for(self, entry: Entry) -> int:
        if self.limit_tokens is None:
            return entry.cost
        return min(entry.cost, self.limit_tokens)

    def _tenant_cap(self) -> int | None:
        if self.max_tenant_share is None or self.limit_tokens is None:
            return None
        return max(1, int(self.limit_tokens * self.max_tenant_share))

    def _enqueue(self, entry: Entry) -> None:
        flow = (entry.tenant, entry.level)
        queue = self._flows.get(flow)
        if queue is None:
            queue = self._flows[flow] = deque()
            self._flow_tokens[flow] = 0
        queue.append(entry)
        self._flow_tokens[flow] += entry.cost
        self.queued_requests += 1
        self.queued_tokens += entry.cost
        if math.isfinite(entry.deadline):
            heapq.heappush(self._deadlines, (entry.deadline, entry.id, entry))

    def _remove(self, entry: Entry) -> None:
        flow = (entry.tenant, entry.level)
        queue = self._flows[flow]
        if queue[0] is entry:
            queue.popleft()
        else:
            queue.remove(entry)
        self._flow_tokens[flow] -= entry.cost
        if not queue:
            del self._flows[flow]
            del self._flow_tokens[flow]
        self.queued_requests -= 1
        self.queued_tokens -= entry.cost

    def _mark_shed(
        self,
        entry: Entry,
        reason: ShedReason,
        now: float,
        effects: Effects,
        *,
        wait: float | None = None,
        retry_after: float | None = None,
    ) -> None:
        entry.state = EntryState.SHED
        entry.shed_reason = reason
        if retry_after is None:
            if wait is None:
                wait = self._hypothetical_wait(entry)
            budget = max(0.0, entry.deadline - entry.enqueued_at)
            if wait is None:
                retry_after = self.retry_after_min
            else:
                # A retry is worthwhile once the backlog has drained to within one queue
                # budget of this request.
                retry_after = wait - budget
            retry_after = min(self.retry_after_max, max(self.retry_after_min, retry_after))
        entry.retry_after = retry_after
        effects.shed.append(entry)

    def _hypothetical_wait(self, entry: Entry) -> float | None:
        """Predicted wait for a request like ``entry`` arriving now (entry not queued)."""
        return self.predicted_wait(entry)

    def _effective_level(self, entry: Entry, now: float) -> int:
        if entry.level <= 0 or not entry.aging_s > 0 or math.isinf(entry.aging_s):
            return entry.level
        return max(0, entry.level - int((now - entry.enqueued_at) / entry.aging_s))

    def _eligible(self, entry: Entry, cap: int | None) -> bool:
        if cap is None:
            return True
        if entry.tenant not in self._tenant_requests:
            return True  # a tenant with nothing in flight may always run one request
        return self._tenant_inflight.get(entry.tenant, 0) + self._charge_for(entry) <= cap

    def _select(self, now: float) -> Entry | None:
        cap = self._tenant_cap()
        best: Entry | None = None
        best_key: tuple[int, float, int] | None = None
        v = self._v
        vtime = self._vtime
        for queue in self._flows.values():
            head = queue[0]
            if not self._eligible(head, cap):
                continue
            start = vtime.get(head.tenant, 0.0)
            key = (self._effective_level(head, now), start if start > v else v, head.id)
            if best_key is None or key < best_key:
                best, best_key = head, key
        return best

    def _fits(self, entry: Entry) -> bool:
        if self.limit_requests is not None and self.inflight_requests >= self.limit_requests:
            return False
        if self.limit_tokens is None:
            return True
        return self.inflight_tokens + self._charge_for(entry) <= self.limit_tokens

    def _dispatch(self, now: float, effects: Effects) -> None:
        while self._flows:
            entry = self._select(now)
            if entry is None or not self._fits(entry):
                return
            self._remove(entry)
            self._admit(entry, now)
            effects.admitted.append(entry)

    def _admit(self, entry: Entry, now: float) -> None:
        entry.state = EntryState.ADMITTED
        entry.admitted_at = now
        entry.charged = self._charge_for(entry)
        self.inflight_tokens += entry.charged
        self.inflight_requests += 1
        self._tenant_inflight[entry.tenant] = (
            self._tenant_inflight.get(entry.tenant, 0) + entry.charged
        )
        self._tenant_requests[entry.tenant] = self._tenant_requests.get(entry.tenant, 0) + 1
        # Advance virtual time.  V follows the smallest start tag among backlogged tenants
        # (including the one just chosen) so a high-priority pick cannot drag V past
        # tenants that are still waiting at a lower level.
        v = self._v
        start = max(v, self._vtime.get(entry.tenant, 0.0))
        low = start
        for tenant, _level in self._flows:
            other = max(v, self._vtime.get(tenant, 0.0))
            if other < low:
                low = other
        self._v = low
        self._vtime[entry.tenant] = start + entry.cost / entry.weight

    def _expire(self, now: float, effects: Effects) -> None:
        heap = self._deadlines
        while heap and (heap[0][0] <= now or heap[0][2].state is not EntryState.QUEUED):
            _, _, entry = heapq.heappop(heap)
            if entry.state is EntryState.QUEUED:
                self._remove(entry)
                self._mark_shed(entry, "queue_timeout", now, effects)

    def _eviction_victim(self, newcomer: Entry) -> Entry:
        """Pick the entry to drop when the queue is over its depth limit.

        The newest entry of the worst priority level goes, so a full queue of batch work
        cannot lock out interactive arrivals.  With a single level that is the newcomer.
        """
        victim = newcomer
        for queue in self._flows.values():
            tail = queue[-1]
            if (tail.level, tail.id) > (victim.level, victim.id):
                victim = tail
        return victim

    def _work_ahead(self, entry: Entry) -> tuple[float, float]:
        """Tokens and requests expected to be admitted before ``entry``."""
        own_flow = (entry.tenant, entry.level)
        own_tokens = 0
        own_requests = 0
        queue = self._flows.get(own_flow)
        if queue is not None:
            for other in queue:
                if other is entry or other.id > entry.id:
                    break
                own_tokens += other.cost
                own_requests += 1
        v = self._v
        start = max(v, self._vtime.get(entry.tenant, 0.0)) + own_tokens / entry.weight
        tokens = float(own_tokens)
        requests = float(own_requests)
        for flow, flow_queue in self._flows.items():
            tenant, level = flow
            if flow == own_flow or level > entry.level:
                continue
            flow_tokens = self._flow_tokens[flow]
            if level < entry.level:
                share = 1.0
            else:
                # Same level, other tenant: it is served while its start tag is below ours.
                head = flow_queue[0]
                other_start = max(v, self._vtime.get(tenant, 0.0))
                allowed = max(0.0, (start - other_start) * head.weight)
                if other_start <= start:
                    allowed += head.cost  # the head whose start tag ties or precedes ours
                share = min(1.0, allowed / flow_tokens) if flow_tokens > 0 else 1.0
            tokens += flow_tokens * share
            requests += len(flow_queue) * share
        return tokens, requests

    def _maybe_prune(self) -> None:
        """Forget tenants with nothing queued and nothing in flight.

        An idle tenant re-enters at the global virtual time anyway; all that is lost is a
        debt of at most one request, so the table stays bounded by the active tenants.
        """
        if len(self._vtime) < _PRUNE_THRESHOLD:
            return
        active = {tenant for tenant, _ in self._flows}
        active.update(self._tenant_requests)
        self._vtime = {tenant: vt for tenant, vt in self._vtime.items() if tenant in active}

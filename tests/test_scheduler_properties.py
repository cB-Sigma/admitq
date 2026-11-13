"""Property-based tests of the scheduler invariants."""

from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from admitq.scheduler import Entry, EntryState, Scheduler

TENANTS = ["a", "b", "c", "d"]

operations = st.lists(
    st.one_of(
        st.tuples(
            st.just("offer"),
            st.sampled_from(TENANTS),
            st.integers(0, 400),  # cost
            st.integers(0, 2),  # priority level
            st.floats(0.0, 5.0),  # queue budget
            st.sampled_from([0.5, 1.0, 2.0, 4.0]),  # weight
        ),
        st.tuples(
            st.just("release"), st.integers(0, 10_000), st.one_of(st.none(), st.integers(0, 800))
        ),
        st.tuples(st.just("cancel"), st.integers(0, 10_000)),
        st.tuples(st.just("advance"), st.floats(0.0, 3.0)),
        st.tuples(st.just("expire")),
    ),
    max_size=120,
)

configs = st.fixed_dictionaries(
    {
        "limit_tokens": st.one_of(st.none(), st.integers(1, 1000)),
        "limit_requests": st.one_of(st.none(), st.integers(1, 6)),
        "max_queue_depth": st.one_of(st.none(), st.integers(0, 8)),
        "max_tenant_share": st.one_of(st.none(), st.sampled_from([0.25, 0.5, 1.0])),
        "oversize": st.sampled_from(["clamp", "reject"]),
        "early_shed": st.booleans(),
    }
)


class Model:
    """Shadow book-keeping the scheduler is checked against after every operation."""

    def __init__(self, scheduler: Scheduler) -> None:
        self.s = scheduler
        self.entries: list[Entry] = []
        self.admitted_seen: dict[int, int] = {}
        self.shed_seen: dict[int, int] = {}
        self.limits_changed = False

    def absorb(self, effects, now: float) -> None:
        for entry in effects.admitted:
            self.admitted_seen[entry.id] = self.admitted_seen.get(entry.id, 0) + 1
            # Queue time never exceeds the bound for admitted requests.
            assert entry.admitted_at == now
            assert entry.admitted_at <= entry.deadline or entry.admitted_at == entry.enqueued_at
            assert entry.queue_time <= max(0.0, entry.deadline - entry.enqueued_at) + 1e-9
        for entry in effects.shed:
            self.shed_seen[entry.id] = self.shed_seen.get(entry.id, 0) + 1
            assert entry.state is EntryState.SHED
            assert entry.shed_reason is not None
            assert entry.retry_after is not None
            assert entry.retry_after >= 0

    def check(self, now: float) -> None:
        s = self.s
        admitted = [e for e in self.entries if e.state is EntryState.ADMITTED]
        queued = [e for e in self.entries if e.state is EntryState.QUEUED]
        # Accounting matches reality.
        assert s.inflight_tokens == sum(e.charged for e in admitted)
        assert s.inflight_requests == len(admitted)
        assert s.queued_requests == len(queued)
        assert s.queued_tokens == sum(e.cost for e in queued)
        # Capacity is never exceeded.
        if s.limit_tokens is not None:
            assert s.inflight_tokens <= s.limit_tokens
        if s.limit_requests is not None:
            assert s.inflight_requests <= s.limit_requests
        if s.max_queue_depth is not None:
            assert s.queued_requests <= s.max_queue_depth
        # Per-tenant share: above the cap only via the "one request always allowed" rule.
        cap = s._tenant_cap()
        if cap is not None:
            for tenant in TENANTS:
                held = [e for e in admitted if e.tenant == tenant]
                assert sum(e.charged for e in held) <= cap or len(held) == 1
        # Nobody is admitted or shed twice, and never both.
        assert all(count == 1 for count in self.admitted_seen.values())
        assert all(count == 1 for count in self.shed_seen.values())
        assert not set(self.admitted_seen) & set(self.shed_seen)
        for entry in self.entries:
            if entry.state in (EntryState.ADMITTED, EntryState.RELEASED):
                assert entry.id in self.admitted_seen
            if entry.state is EntryState.SHED:
                assert entry.id in self.shed_seen
            if entry.state is EntryState.QUEUED:
                assert entry.deadline > now  # expired entries never linger
        # No idle capacity while an admissible head is waiting.
        head = s._select(now)
        assert head is None or not s._fits(head)


@given(config=configs, ops=operations)
def test_invariants_hold_under_arbitrary_operation_sequences(config, ops):
    s = Scheduler(**config)
    model = Model(s)
    now = 0.0
    for op in ops:
        kind = op[0]
        if kind == "offer":
            _, tenant, cost, level, budget, weight = op
            entry, effects = s.offer(
                tenant=tenant,
                cost=cost,
                now=now,
                deadline=now + budget,
                level=level,
                weight=weight,
                aging_s=max(budget / 2, 1e-3),
            )
            model.entries.append(entry)
            model.absorb(effects, now)
            assert entry.state in (EntryState.ADMITTED, EntryState.QUEUED, EntryState.SHED)
        elif kind == "release" and model.entries:
            entry = model.entries[op[1] % len(model.entries)]
            was_admitted = entry.state is EntryState.ADMITTED
            effects = s.release(entry, now, actual_cost=op[2])
            if not was_admitted:
                # Releasing twice (or releasing a non-holder) does nothing at all - not
                # even lazy expiry, so there is nothing new to check.
                assert not effects
                continue
            model.absorb(effects, now)
        elif kind == "cancel" and model.entries:
            entry = model.entries[op[1] % len(model.entries)]
            live = entry.state in (EntryState.QUEUED, EntryState.ADMITTED)
            effects = s.cancel(entry, now)
            if not live:
                assert not effects  # cancelling something already finished is a no-op
                continue
            model.absorb(effects, now)
        elif kind == "advance":
            now += op[1]
            continue  # time passing alone changes nothing until the next call
        elif kind == "expire":
            model.absorb(s.expire(now), now)
        model.check(now)

    # Drain: release everything that holds capacity until nothing is left.  The queue must
    # empty without help from the clock (no deadlock) and all capacity must come back.
    for _ in range(len(model.entries) + 1):
        holders = [e for e in model.entries if e.state is EntryState.ADMITTED]
        if not holders:
            break
        for entry in holders:
            model.absorb(s.release(entry, now), now)
        model.check(now)
    assert s.inflight_tokens == 0
    assert s.inflight_requests == 0
    assert s.queued_requests == 0
    assert s.queued_tokens == 0
    assert not s._tenant_inflight
    assert not s._tenant_requests
    assert not s._flows
    assert all(
        e.state in (EntryState.RELEASED, EntryState.SHED, EntryState.CANCELLED)
        for e in model.entries
    )


@given(
    weights=st.lists(st.sampled_from([0.5, 1.0, 2.0, 3.0, 8.0]), min_size=2, max_size=5),
    costs=st.lists(st.integers(1, 500), min_size=40, max_size=200),
    rounds=st.integers(10, 150),
)
def test_weighted_fair_queuing_bounds_unfairness(weights, costs, rounds):
    """Start-time fair queuing guarantee: for any two tenants that stay backlogged, the
    difference in weight-normalised service never exceeds one maximum-size request each.
    In particular no backlogged tenant starves, whatever the others send."""
    s = Scheduler(limit_requests=1, early_shed=False)
    running, _ = s.offer(tenant="warmup", cost=1, now=0.0, deadline=math.inf)
    tenants = [f"t{i}" for i in range(len(weights))]
    max_cost = dict.fromkeys(tenants, 0)
    # Every tenant gets far more queued work than can be served: all stay backlogged.
    for index in range(rounds + 1):
        for position, tenant in enumerate(tenants):
            cost = costs[(index * len(tenants) + position) % len(costs)]
            max_cost[tenant] = max(max_cost[tenant], cost)
            s.offer(tenant=tenant, cost=cost, now=0.0, deadline=math.inf, weight=weights[position])
    served = dict.fromkeys(tenants, 0.0)
    for step in range(rounds):
        effects = s.release(running, now=float(step + 1))
        assert len(effects.admitted) == 1
        running = effects.admitted[0]
        served[running.tenant] += running.cost
        for i, a in enumerate(tenants):
            for j, b in enumerate(tenants):
                if i < j:
                    gap = abs(served[a] / weights[i] - served[b] / weights[j])
                    bound = max_cost[a] / weights[i] + max_cost[b] / weights[j]
                    assert gap <= bound + 1e-6


@given(
    noisy_backlog=st.integers(1, 400),
    noisy_cost=st.integers(1, 1000),
    quiet_cost=st.integers(1, 1000),
    limit=st.integers(1000, 4000),
)
def test_a_newly_arriving_tenant_waits_at_most_two_dispatches(
    noisy_backlog, noisy_cost, quiet_cost, limit
):
    """However deep one tenant's backlog, another tenant's first request is among the next
    two admitted (it can lose one tie to an entry that is already at the head)."""
    s = Scheduler(limit_tokens=limit, limit_requests=1, early_shed=False)
    running, _ = s.offer(tenant="noisy", cost=noisy_cost, now=0.0, deadline=math.inf)
    for _ in range(noisy_backlog):
        s.offer(tenant="noisy", cost=noisy_cost, now=0.0, deadline=math.inf)
    quiet, _ = s.offer(tenant="quiet", cost=quiet_cost, now=1.0, deadline=math.inf)
    for step in range(2):
        effects = s.release(running, now=2.0 + step)
        running = effects.admitted[0]
        if running is quiet:
            break
    assert quiet.state is EntryState.ADMITTED


@given(
    arrivals=st.lists(
        st.tuples(st.floats(0.0, 2.0), st.integers(0, 1), st.floats(0.5, 20.0)),
        min_size=1,
        max_size=60,
    ),
    aging=st.floats(0.5, 5.0),
)
def test_ageing_bounds_how_long_low_priority_work_can_be_overtaken(arrivals, aging):
    """Starvation protection: once a batch request has waited ``aging_s`` it competes as
    interactive, so no interactive request that arrives *after* that point overtakes it."""
    s = Scheduler(limit_requests=1, early_shed=False)
    now = 0.0
    running, _ = s.offer(tenant="x", cost=1, now=now, deadline=math.inf)
    batch, _ = s.offer(tenant="batch", cost=1, now=now, deadline=math.inf, level=1, aging_s=aging)
    admitted_before_batch: list[Entry] = []
    for gap, _unused, hold in arrivals:
        now += gap
        s.offer(tenant="inter", cost=1, now=now, deadline=math.inf, level=0)
        now += hold
        effects = s.release(running, now)
        running = effects.admitted[0]
        if running is batch:
            break
        admitted_before_batch.append(running)
    for entry in admitted_before_batch:
        assert entry.enqueued_at < batch.enqueued_at + aging

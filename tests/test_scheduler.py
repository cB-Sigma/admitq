"""Deterministic tests of the scheduler state machine, driven by explicit timestamps."""

from __future__ import annotations

import math

import pytest

from admitq.scheduler import EntryState, Scheduler

A, Q, S = EntryState.ADMITTED, EntryState.QUEUED, EntryState.SHED


def offer(s: Scheduler, tenant: str = "t", cost: int = 100, now: float = 0.0, **kw):
    kw.setdefault("deadline", now + 2.0)
    return s.offer(tenant=tenant, cost=cost, now=now, **kw)


def test_admits_until_token_limit_then_queues():
    s = Scheduler(limit_tokens=250)
    a, _ = offer(s)
    b, _ = offer(s)
    c, _ = offer(s)
    assert (a.state, b.state, c.state) == (A, A, Q)
    assert s.inflight_tokens == 200
    assert s.queued_requests == 1
    assert s.queued_tokens == 100
    effects = s.release(a, now=0.5)
    assert effects.admitted == [c]
    assert c.queue_time == pytest.approx(0.5)
    assert s.inflight_tokens == 200
    assert s.queued_requests == 0


def test_request_limit_is_independent_of_tokens():
    s = Scheduler(limit_requests=2)
    entries = [offer(s, cost=10_000)[0] for _ in range(3)]
    assert [e.state for e in entries] == [A, A, Q]
    assert s.inflight_requests == 2


def test_no_limits_admits_everything():
    s = Scheduler()
    assert all(offer(s)[0].state is A for _ in range(100))
    assert s.predicted_wait(offer(s)[0]) is None


def test_release_is_idempotent():
    s = Scheduler(limit_tokens=100)
    a, _ = offer(s)
    s.release(a, now=1.0)
    assert s.inflight_tokens == 0
    assert not s.release(a, now=2.0)
    assert s.inflight_tokens == 0
    assert s.inflight_requests == 0
    assert a.state is EntryState.RELEASED


def test_queue_deadline_sheds_with_queue_timeout():
    s = Scheduler(limit_tokens=100, early_shed=False)
    a, _ = offer(s)
    b, _ = offer(s, now=0.0, deadline=2.0)
    assert s.next_deadline() == 2.0
    assert not s.expire(now=1.999)
    effects = s.expire(now=2.0)
    assert effects.shed == [b]
    assert b.shed_reason == "queue_timeout"
    assert b.retry_after is not None
    assert b.retry_after >= 1.0
    assert s.queued_requests == 0
    assert s.next_deadline() is None
    assert a.state is A


def test_never_admits_after_the_deadline_even_without_a_timer():
    # No expire() call: the release at t=5 must shed the stale entry, not admit it.
    s = Scheduler(limit_tokens=100, early_shed=False)
    a, _ = offer(s)
    b, _ = offer(s, deadline=2.0)
    effects = s.release(a, now=5.0)
    assert effects.admitted == []
    assert effects.shed == [b]


def test_zero_deadline_never_queues():
    s = Scheduler(limit_tokens=100)
    offer(s)
    b, effects = offer(s, deadline=0.0)
    assert b.state is S
    assert b.shed_reason == "queue_timeout"
    assert effects.shed == [b]


def test_queue_depth_limit():
    s = Scheduler(limit_tokens=100, max_queue_depth=2, early_shed=False)
    offer(s)
    q1, _ = offer(s)
    q2, _ = offer(s)
    q3, effects = offer(s)
    assert (q1.state, q2.state, q3.state) == (Q, Q, S)
    assert q3.shed_reason == "queue_full"
    assert effects.shed == [q3]


def test_full_queue_evicts_lower_priority_for_higher():
    s = Scheduler(limit_tokens=100, max_queue_depth=2, early_shed=False)
    offer(s)
    b1, _ = offer(s, level=1)
    b2, _ = offer(s, level=1)
    i1, effects = offer(s, level=0)
    assert i1.state is Q
    assert effects.shed == [b2]  # newest of the worst level
    assert b2.shed_reason == "queue_full"
    assert b1.state is Q


def test_oversize_clamp_runs_alone():
    s = Scheduler(limit_tokens=100)
    big, _ = offer(s, cost=1000)
    assert big.state is A
    assert big.charged == 100
    assert s.inflight_tokens == 100
    small, _ = offer(s, cost=1)
    assert small.state is Q
    s.release(big, now=1.0)
    assert small.state is A


def test_oversize_reject():
    s = Scheduler(limit_tokens=100, oversize="reject")
    big, effects = offer(s, cost=101)
    assert big.state is S
    assert big.shed_reason == "too_large"
    assert effects.shed == [big]
    assert s.queued_requests == 0


def test_cancel_queued_and_admitted():
    s = Scheduler(limit_tokens=100, early_shed=False)
    a, _ = offer(s)
    b, _ = offer(s)
    c, _ = offer(s)
    s.cancel(b, now=0.1)
    assert b.state is EntryState.CANCELLED
    assert s.queued_requests == 1
    effects = s.cancel(a, now=0.2)  # cancelling an admitted entry releases it
    assert effects.admitted == [c]
    assert a.state is EntryState.RELEASED
    assert not s.cancel(b, now=0.3)


def test_head_of_line_is_strict():
    # A big request at the head must not be overtaken by small ones behind it.
    s = Scheduler(limit_tokens=100, early_shed=False)
    a, _ = offer(s, cost=60)
    big, _ = offer(s, cost=80)
    small, _ = offer(s, cost=10)
    assert (big.state, small.state) == (Q, Q)
    assert s.inflight_tokens == 60  # 40 free, but the head needs 80
    s.release(a, now=1.0)
    assert (big.state, small.state) == (A, A)


def test_priority_levels_order_dispatch():
    s = Scheduler(limit_requests=1, early_shed=False)
    first, _ = offer(s)
    batch, _ = offer(s, level=1, deadline=100.0)
    inter, _ = offer(s, level=0, deadline=100.0)
    effects = s.release(first, now=1.0)
    assert effects.admitted == [inter]
    effects = s.release(inter, now=2.0)
    assert effects.admitted == [batch]


def test_ageing_promotes_waiting_batch_work():
    s = Scheduler(limit_requests=1, early_shed=False)
    running, _ = offer(s)
    batch, _ = offer(s, level=1, deadline=100.0, aging_s=5.0)
    # Interactive arrivals keep coming; before 5 s they always win.
    i1, _ = offer(s, now=1.0, level=0, deadline=100.0)
    effects = s.release(running, now=2.0)
    assert effects.admitted == [i1]
    i2, _ = offer(s, now=3.0, level=0, deadline=100.0)
    # At t=6 the batch entry has aged to level 0 and is older than i2: it goes first.
    effects = s.release(i1, now=6.0)
    assert effects.admitted == [batch]
    assert i2.state is Q


def test_strict_priority_when_ageing_disabled():
    s = Scheduler(limit_requests=1, early_shed=False)
    running, _ = offer(s)
    batch, _ = offer(s, level=1, deadline=1000.0, aging_s=math.inf)
    for step in range(1, 50):
        nxt, _ = offer(s, now=float(step), level=0, deadline=1000.0)
        effects = s.release(running, now=float(step))
        assert effects.admitted == [nxt]
        running = nxt
    assert batch.state is Q


def test_weighted_fair_queuing_shares_follow_weights():
    s = Scheduler(limit_requests=1, early_shed=False)
    running, _ = offer(s, tenant="warm")
    for _ in range(300):
        offer(s, tenant="a", weight=1.0, deadline=math.inf)
        offer(s, tenant="b", weight=3.0, deadline=math.inf)
    served = {"a": 0, "b": 0}
    for step in range(200):
        effects = s.release(running, now=float(step))
        (running,) = effects.admitted
        served[running.tenant] += 1
    assert served["a"] + served["b"] == 200
    assert served["b"] == pytest.approx(150, abs=2)


def test_fair_queuing_is_by_tokens_not_requests():
    s = Scheduler(limit_requests=1, early_shed=False)
    running, _ = offer(s, tenant="warm")
    for _ in range(400):
        offer(s, tenant="big", cost=400, deadline=math.inf)
        offer(s, tenant="small", cost=100, deadline=math.inf)
    tokens = {"big": 0, "small": 0}
    for step in range(250):
        effects = s.release(running, now=float(step))
        (running,) = effects.admitted
        tokens[running.tenant] += running.cost
    # Equal weights: equal token service, i.e. four small requests per big one.
    assert abs(tokens["big"] - tokens["small"]) <= 400


def test_burst_cannot_starve_a_later_tenant():
    s = Scheduler(limit_requests=1, early_shed=False)
    running, _ = offer(s, tenant="noisy")
    for _ in range(1000):
        offer(s, tenant="noisy", deadline=math.inf)
    late, _ = offer(s, tenant="quiet", now=1.0, deadline=math.inf)
    effects = s.release(running, now=2.0)
    # The quiet tenant's first request is served next, ahead of 1000 queued noisy ones.
    assert effects.admitted == [late]


def test_idle_tenant_gets_no_credit_for_the_past():
    s = Scheduler(limit_requests=1, early_shed=False)
    running, _ = offer(s, tenant="busy")
    # "busy" is served alone for a long time.
    for step in range(100):
        nxt, _ = offer(s, tenant="busy", now=float(step), deadline=math.inf)
        s.release(running, now=float(step))
        running = nxt
    # Now both tenants are backlogged: service must alternate, not favour the newcomer
    # for the next 100 rounds.
    for _ in range(20):
        offer(s, tenant="busy", now=100.0, deadline=math.inf)
        offer(s, tenant="idle", now=100.0, deadline=math.inf)
    order = []
    for step in range(20):
        effects = s.release(running, now=101.0 + step)
        (running,) = effects.admitted
        order.append(running.tenant)
    assert order.count("idle") == pytest.approx(10, abs=1)


def test_actual_cost_corrects_fairness_account():
    s = Scheduler(limit_requests=1, early_shed=False)
    running, _ = offer(s, tenant="a", cost=1000)  # estimated 1000 ...
    for _ in range(10):
        offer(s, tenant="a", cost=100, deadline=math.inf)
        offer(s, tenant="b", cost=100, deadline=math.inf)
    # ... but it really used 100, so "a" is not penalised for the over-estimate.
    order = []
    effects = s.release(running, now=1.0, actual_cost=100)
    for step in range(8):
        (running,) = effects.admitted
        order.append(running.tenant)
        effects = s.release(running, now=2.0 + step)
    assert order.count("a") == 4


def test_max_tenant_share_caps_one_tenant():
    s = Scheduler(limit_tokens=1000, max_tenant_share=0.5, early_shed=False)
    greedy = [offer(s, tenant="greedy", cost=100, deadline=math.inf)[0] for _ in range(8)]
    assert sum(e.state is A for e in greedy) == 5
    assert s.tenant_inflight_tokens("greedy") == 500
    # Someone else is admitted immediately although "greedy" has entries queued ahead.
    other, _ = offer(s, tenant="other", cost=400)
    assert other.state is A
    # Freed capacity goes back to the capped tenant only up to its share.
    s.release(greedy[0], now=1.0)
    assert sum(e.state is A for e in greedy) == 5
    assert s.tenant_inflight_tokens("greedy") == 500


def test_tenant_share_always_allows_one_request():
    s = Scheduler(limit_tokens=1000, max_tenant_share=0.1)
    big, _ = offer(s, tenant="t", cost=900)
    assert big.state is A


def test_predicted_wait_uses_measured_drain_rate():
    s = Scheduler(limit_tokens=100, early_shed=False)
    a, _ = offer(s, cost=100)
    queued, _ = offer(s, cost=100, deadline=math.inf)
    assert s.predicted_wait(queued) is None  # nothing has completed yet
    s.release(a, now=2.0)  # 100 tokens held for 2 s -> drains 50 tokens/s
    s.release(queued, now=4.0)
    b, _ = offer(s, cost=100, now=4.0)
    c, _ = offer(s, cost=100, now=4.0, deadline=math.inf)
    d, _ = offer(s, cost=100, now=4.0, deadline=math.inf)
    assert b.state is A
    assert s.predicted_wait(c) == pytest.approx(2.0)
    assert s.predicted_wait(d) == pytest.approx(4.0)


def test_early_shed_when_predicted_wait_exceeds_deadline():
    s = Scheduler(limit_tokens=100)
    a, _ = offer(s)
    s.release(a, now=2.0)  # hold time 2 s
    b, _ = offer(s, now=2.0)
    ok, _ = offer(s, now=2.0, deadline=2.0 + 2.5)  # predicted 2 s < 2.5 s budget
    assert ok.state is Q
    late, effects = offer(s, now=2.0, deadline=2.0 + 2.5)  # predicted 4 s > 2.5 s
    assert late.state is S
    assert late.shed_reason == "predicted_wait"
    assert effects.shed == [late]
    # Honest Retry-After: predicted wait minus what the queue would have absorbed.
    assert late.retry_after == pytest.approx(4.0 - 2.5)
    assert b.state is A


def test_early_shed_respects_fair_queuing():
    # A quiet tenant must not be shed just because a noisy one has a long backlog.
    s = Scheduler(limit_tokens=100, max_queue_depth=None)
    a, _ = offer(s, tenant="noisy")
    s.release(a, now=1.0)  # hold time 1 s
    running, _ = offer(s, tenant="noisy", now=1.0)
    for _ in range(50):
        offer(s, tenant="noisy", now=1.0, deadline=1000.0)
    quiet, _ = offer(s, tenant="quiet", now=1.0, deadline=1.0 + 3.0)
    assert quiet.state is Q
    assert s.predicted_wait(quiet) <= 2.0
    noisy_late, _ = offer(s, tenant="noisy", now=1.0, deadline=1.0 + 3.0)
    assert noisy_late.state is S
    effects = s.release(running, now=2.0)
    assert quiet in effects.admitted or s.release(effects.admitted[0], now=3.0).admitted == [quiet]


def test_retry_after_is_clamped():
    s = Scheduler(limit_tokens=100, retry_after_min=2.0, retry_after_max=10.0)
    a, _ = offer(s)
    s.release(a, now=1000.0)  # enormous hold time
    offer(s, now=1000.0)
    late, _ = offer(s, now=1000.0, deadline=1001.0)
    assert late.state is S
    assert late.retry_after == 10.0


def test_set_limits_grow_and_shrink():
    s = Scheduler(limit_tokens=100, early_shed=False)
    a, _ = offer(s)
    b, _ = offer(s, deadline=math.inf)
    effects = s.set_limits(0.0, limit_tokens=200)
    assert effects.admitted == [b]
    s.set_limits(0.0, limit_tokens=50)
    c, _ = offer(s, deadline=math.inf)
    assert c.state is Q
    s.release(a, now=1.0)
    assert c.state is Q  # still 100 in flight > 50
    s.release(b, now=1.0)
    assert c.state is A
    assert c.charged == 50  # clamped to the new limit
    with pytest.raises(ValueError, match="limit_tokens"):
        s.set_limits(0.0, limit_tokens=0)


def test_snapshot_and_validation():
    s = Scheduler(limit_tokens=100, limit_requests=4)
    a, _ = offer(s, cost=30)
    snap = s.snapshot()
    assert (snap.inflight_tokens, snap.inflight_requests) == (30, 1)
    assert (snap.limit_tokens, snap.limit_requests) == (100, 4)
    assert snap.mean_hold_s is None
    s.release(a, now=3.0)
    assert s.snapshot().mean_hold_s == pytest.approx(3.0)
    with pytest.raises(ValueError, match="cost"):
        offer(s, cost=-1)
    with pytest.raises(ValueError, match="weight"):
        offer(s, weight=0)
    for kwargs in (
        {"limit_tokens": 0},
        {"limit_requests": 0},
        {"max_queue_depth": -1},
        {"max_tenant_share": 0.0},
        {"max_tenant_share": 1.5},
    ):
        with pytest.raises(ValueError):  # noqa: PT011
            Scheduler(**kwargs)


def test_many_tenants_do_not_leak_state():
    s = Scheduler(limit_requests=1)
    for index in range(10_000):
        entry, _ = offer(s, tenant=f"t{index}", now=float(index))
        s.release(entry, now=float(index) + 0.5)
    assert len(s._vtime) < 5000
    assert not s._flows
    assert not s._tenant_inflight

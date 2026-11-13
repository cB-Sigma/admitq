"""The programmatic API, on a virtual-time event loop."""

from __future__ import annotations

import asyncio
import math

import pytest

from admitq import (
    AdmissionConfig,
    AdmissionController,
    ConfigError,
    Cost,
    PriorityClass,
    Shed,
    TenantLimits,
)
from admitq.backends import MemoryBackend, Reservation

from .helpers import settle, virtual


@virtual
async def test_admit_context_manager_holds_and_returns_capacity():
    c = AdmissionController(max_inflight_tokens=1000)
    async with c.admit(tenant="acme", cost=Cost(300, 100)) as ticket:
        assert c.snapshot().inflight_tokens == 400
        assert c.snapshot().inflight_requests == 1
        assert ticket.tenant == "acme"
        assert ticket.priority == "interactive"
        assert ticket.cost.total == 400
        assert ticket.queue_time == 0.0
        assert ticket.remaining_tokens is None
        assert not ticket.released
        assert ticket.actual_tokens is None
    assert ticket.released
    assert ticket.outcome == "ok"
    assert ticket.actual_tokens == 400
    assert c.snapshot().inflight_tokens == 0
    assert c.metrics.value("admitq_admitted_total", tenant="acme", priority="interactive") == 1
    assert c.metrics.value("admitq_completed_total", tenant="acme", outcome="ok") == 1


@virtual
async def test_plain_int_cost_and_estimate_helper():
    c = AdmissionController(max_inflight_tokens=1000)
    ticket = await c.acquire(cost=250)
    assert ticket.cost == Cost(250, 0)
    assert ticket.release() is True
    assert ticket.release() is False  # idempotent
    body = {"messages": [{"role": "user", "content": "x" * 400}], "max_tokens": 50}
    assert c.estimate(body) == Cost(104, 50)
    assert c.estimate({"input": "x" * 400}, "embeddings") == Cost(100, 0)


@virtual
async def test_waits_in_queue_until_capacity_is_released():
    c = AdmissionController(max_inflight_tokens=100, max_queue_s=5.0)
    loop = asyncio.get_running_loop()
    first = await c.acquire(cost=100)
    waiter = asyncio.create_task(c.acquire(cost=100))
    await settle()
    assert not waiter.done()
    assert c.snapshot().queued_requests == 1
    await asyncio.sleep(1.5)
    first.release()
    second = await waiter
    assert second.queue_time == pytest.approx(1.5)
    assert loop.time() == pytest.approx(1.5)
    second.release()
    assert c.metrics.value("admitq_queued_total", tenant="default", priority="interactive") == 1
    assert c.metrics.histogram("admitq_queue_time_seconds", priority="interactive")[0] == 2


@virtual
async def test_queue_timeout_sheds_at_exactly_max_queue_s():
    c = AdmissionController(max_inflight_tokens=100, max_queue_s=2.0, early_shed=False)
    loop = asyncio.get_running_loop()
    holder = await c.acquire(cost=100)
    with pytest.raises(Shed) as info:
        await c.acquire(cost=100)
    assert loop.time() == pytest.approx(2.0)
    shed = info.value
    assert shed.reason == "queue_timeout"
    assert shed.status == 503
    assert shed.retry_after >= 1.0
    assert shed.body()["error"]["code"] == "server_overloaded"
    assert shed.body()["error"]["type"] == "server_error"
    assert shed.headers()["x-admitq-shed-reason"] == "queue_timeout"
    assert int(shed.headers()["retry-after"]) >= 1
    assert c.snapshot().queued_requests == 0
    assert c.metrics.total("admitq_shed_total", reason="queue_timeout") == 1
    holder.release()
    assert c.snapshot().inflight_tokens == 0


@virtual
async def test_caller_timeout_shortens_the_queue_deadline():
    c = AdmissionController(max_inflight_tokens=100, max_queue_s=10.0, early_shed=False)
    loop = asyncio.get_running_loop()
    await c.acquire(cost=100)
    with pytest.raises(Shed):
        await c.acquire(cost=100, timeout=0.75)
    assert loop.time() == pytest.approx(0.75)


@virtual
async def test_predicted_wait_sheds_immediately_with_honest_retry_after():
    c = AdmissionController(max_inflight_tokens=100, max_queue_s=1.0)
    loop = asyncio.get_running_loop()
    ticket = await c.acquire(cost=100)
    await asyncio.sleep(4.0)
    ticket.release()  # the controller has now seen: 100 tokens take 4 s
    await c.acquire(cost=100)
    started = loop.time()
    with pytest.raises(Shed) as info:
        await c.acquire(cost=100)
    assert loop.time() == started  # refused on arrival, not after waiting
    assert info.value.reason == "predicted_wait"
    assert info.value.retry_after == pytest.approx(4.0 - 1.0)


@virtual
async def test_overload_status_can_be_429():
    c = AdmissionController(max_inflight_requests=1, max_queue_s=0.0, overload_status=429)
    await c.acquire(cost=1)
    with pytest.raises(Shed) as info:
        await c.acquire(cost=1)
    assert info.value.status == 429
    assert info.value.reason == "queue_timeout"


@virtual
async def test_oversize_reject_is_a_400():
    c = AdmissionController(max_inflight_tokens=100, oversize="reject")
    with pytest.raises(Shed) as info:
        await c.acquire(cost=101)
    assert info.value.status == 400
    assert info.value.reason == "too_large"
    assert info.value.body()["error"]["type"] == "invalid_request_error"


@virtual
async def test_tenant_token_budget_sheds_with_exact_retry_after():
    c = AdmissionController(tenant_tpm={"default": 6000})  # 100 tokens/s
    loop = asyncio.get_running_loop()
    ticket = await c.acquire(tenant="a", cost=6000)
    assert ticket.remaining_tokens == 0
    assert ticket.limits == TenantLimits(tpm=6000.0)
    ticket.release()
    with pytest.raises(Shed) as info:
        await c.acquire(tenant="a", cost=250)
    shed = info.value
    assert (shed.status, shed.reason, shed.tenant) == (429, "tenant_tokens", "a")
    assert shed.retry_after == pytest.approx(2.5)
    assert shed.retry_after_header == "3"
    assert shed.body()["error"]["code"] == "rate_limit_exceeded"
    # "default" is a per-tenant default, not a shared bucket.
    (await c.acquire(tenant="b", cost=6000)).release()
    # The promise is exact.
    await asyncio.sleep(shed.retry_after)
    (await c.acquire(tenant="a", cost=250)).release()
    assert loop.time() == pytest.approx(2.5)


@virtual
async def test_tenant_request_budget_and_overrides():
    c = AdmissionController(tenant_rpm={"default": 60, "vip": 6000}, burst_s=1.0)
    (await c.acquire(tenant="a", cost=1)).release()
    with pytest.raises(Shed) as info:
        await c.acquire(tenant="a", cost=1)
    assert info.value.reason == "tenant_requests"
    assert info.value.retry_after == pytest.approx(1.0)
    for _ in range(100):
        (await c.acquire(tenant="vip", cost=1)).release()


@virtual
async def test_bare_number_budget_applies_to_every_tenant():
    c = AdmissionController(tenant_tpm=100)
    (await c.acquire(tenant="a", cost=100)).release()
    (await c.acquire(tenant="b", cost=100)).release()
    with pytest.raises(Shed):
        await c.acquire(tenant="a", cost=100)


@virtual
async def test_usage_correction_refunds_over_estimates():
    c = AdmissionController(tenant_tpm=1000)
    async with c.admit(tenant="t", cost=Cost(100, 900)) as ticket:
        assert ticket.remaining_tokens == 0
        ticket.report_usage(prompt_tokens=120, completion_tokens=80, total_tokens=200)
    assert ticket.actual_tokens == 200
    # 800 of the 1000 reserved tokens came back.
    retry = await c.acquire(tenant="t", cost=800)
    assert retry.remaining_tokens == pytest.approx(0, abs=1e-6)
    assert c.metrics.value("admitq_estimated_tokens_total", tenant="t") == 1800
    assert c.metrics.value("admitq_actual_tokens_total", tenant="t", source="usage") == 200
    assert c.metrics.histogram("admitq_estimate_ratio", source="usage") == (1, 0.2)


@virtual
async def test_usage_correction_charges_under_estimates():
    c = AdmissionController(tenant_tpm=600)  # 10 tokens/s
    async with c.admit(tenant="t", cost=100) as ticket:
        ticket.report_usage(completion_tokens=400)  # prompt falls back to the estimate
    assert ticket.actual_tokens == 500
    with pytest.raises(Shed) as info:
        await c.acquire(tenant="t", cost=200)
    assert info.value.retry_after == pytest.approx(10.0)  # 600 - 500 = 100 left, need 200


@virtual
async def test_report_usage_accepts_an_openai_usage_dict():
    c = AdmissionController(max_inflight_tokens=1000)
    ticket = await c.acquire(cost=500)
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    ticket.report_usage(**usage)
    ticket.release()
    assert ticket.actual_tokens == 15
    other = await c.acquire(cost=500)
    other.report_usage(prompt_tokens=10, total_tokens=30)
    other.report_usage()  # nothing given: keeps the previous report
    other.release()
    assert other.actual_tokens == 30


@virtual
async def test_failed_requests_are_not_charged_tokens():
    c = AdmissionController(tenant_tpm=1000, tenant_rpm=600)
    with pytest.raises(RuntimeError, match="boom"):
        async with c.admit(tenant="t", cost=1000):
            raise RuntimeError("boom")
    ticket = await c.acquire(tenant="t", cost=1000)  # tokens refunded in full
    assert ticket.remaining_requests == 598  # ...but both requests still count
    assert c.metrics.value("admitq_completed_total", tenant="t", outcome="error") == 1
    assert c.snapshot().inflight_requests == 1


@virtual
async def test_budget_is_refunded_when_the_queue_sheds():
    c = AdmissionController(
        max_inflight_requests=1, max_queue_s=1.0, tenant_tpm=1000, tenant_rpm=600, early_shed=False
    )
    holder = await c.acquire(tenant="other", cost=10)
    with pytest.raises(Shed) as info:
        await c.acquire(tenant="t", cost=1000)
    assert info.value.reason == "queue_timeout"
    holder.release()
    ticket = await c.acquire(tenant="t", cost=1000)  # would be a 429 without the refund
    assert ticket.remaining_requests == 599


@virtual
async def test_cancelling_a_queued_acquire_withdraws_it_and_refunds():
    c = AdmissionController(max_inflight_requests=1, max_queue_s=60.0, tenant_tpm=1000)
    holder = await c.acquire(tenant="other", cost=10)
    task = asyncio.create_task(c.acquire(tenant="t", cost=1000))
    await settle()
    assert c.snapshot().queued_requests == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert c.snapshot().queued_requests == 0
    assert c.snapshot().queued_tokens == 0
    holder.release()
    assert c.snapshot().inflight_requests == 0
    (await c.acquire(tenant="t", cost=1000)).release()
    assert c.metrics.value("admitq_completed_total", tenant="t", outcome="abandoned_in_queue") == 1


@virtual
async def test_cancel_racing_with_admission_does_not_leak():
    # The waiter is cancelled and capacity is released in the same loop iteration, in
    # both orders.  Either the waiter got the slot and gave it back, or it never had it.
    for release_first in (True, False):
        c = AdmissionController(max_inflight_requests=1, max_queue_s=60.0)
        holder = await c.acquire(cost=10)
        task = asyncio.create_task(c.acquire(cost=10))
        await settle()
        if release_first:
            holder.release()
            task.cancel()
        else:
            task.cancel()
            holder.release()
        with pytest.raises(asyncio.CancelledError):
            await task
        await settle()
        snap = c.snapshot()
        assert (snap.inflight_requests, snap.inflight_tokens, snap.queued_requests) == (0, 0, 0)


@virtual
async def test_cancelled_context_manager_releases_as_cancelled():
    c = AdmissionController(max_inflight_tokens=100)

    async def work() -> None:
        async with c.admit(cost=100):
            await asyncio.sleep(3600)

    task = asyncio.create_task(work())
    await settle()
    assert c.snapshot().inflight_tokens == 100
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert c.snapshot().inflight_tokens == 0
    assert c.metrics.value("admitq_completed_total", tenant="default", outcome="cancelled") == 1


@virtual
async def test_priorities_and_per_class_queue_deadlines():
    c = AdmissionController(
        max_inflight_requests=1,
        max_queue_s=1.0,
        early_shed=False,
        priorities=[
            PriorityClass("interactive", 0),
            PriorityClass("batch", 1, max_queue_s=30.0, aging_s=math.inf),
        ],
    )
    loop = asyncio.get_running_loop()
    holder = await c.acquire(cost=1)
    order: list[str] = []

    async def run(name: str, priority: str | None) -> None:
        try:
            async with c.admit(cost=1, priority=priority) as ticket:
                order.append(f"{name}:{ticket.priority}")
                await asyncio.sleep(0.1)
        except Shed as shed:
            order.append(f"{name}:shed:{shed.reason}@{loop.time():.1f}")

    tasks = [asyncio.create_task(run("b1", "batch"))]
    await settle()
    tasks.append(asyncio.create_task(run("i1", "interactive")))
    tasks.append(asyncio.create_task(run("i2", None)))  # default class
    tasks.append(asyncio.create_task(run("u1", "no-such-class")))  # unknown -> default
    await settle()
    await asyncio.sleep(0.5)
    holder.release()
    await asyncio.gather(*tasks)
    assert order == ["i1:interactive", "i2:interactive", "u1:interactive", "b1:batch"]


@virtual
async def test_starvation_protection_by_default():
    # Default ageing: a batch request competes as interactive after half its deadline.
    c = AdmissionController(
        max_inflight_requests=1,
        max_queue_s=100.0,
        early_shed=False,
        priorities={"interactive": 0, "batch": 1},
    )
    holder = await c.acquire(cost=1)
    served: list[str] = []

    async def run(name: str, priority: str) -> None:
        try:
            async with c.admit(cost=1, priority=priority):
                served.append(name)
                await asyncio.sleep(10.0)
        except Shed:
            pass  # the tail of the interactive backlog times out after 100 s

    tasks = [asyncio.create_task(run("batch", "batch"))]
    await settle()
    # Plenty of interactive work is queued behind it: 20 requests of 10 s each.
    tasks.extend(asyncio.create_task(run(f"i{index}", "interactive")) for index in range(20))
    await settle()
    holder.release()
    await asyncio.gather(*tasks)
    # Strict priority would serve the batch request last (after 200 s).  With ageing it
    # is promoted at t=50 s (half of its 100 s deadline) and, being the oldest, goes next.
    assert served.index("batch") == 5
    assert c.config.default_priority == "interactive"


@virtual
async def test_fairness_between_tenants_through_the_public_api():
    c = AdmissionController(
        max_inflight_requests=1, max_queue_s=1000.0, early_shed=False, tenant_weights={"gold": 3}
    )
    holder = await c.acquire(cost=1)
    served: list[str] = []

    async def run(tenant: str) -> None:
        async with c.admit(tenant=tenant, cost=100):
            served.append(tenant)
            await asyncio.sleep(1.0)

    tasks = [asyncio.create_task(run(t)) for t in ["gold", "basic"] * 40]
    await settle()
    holder.release()
    await asyncio.sleep(40.5)
    assert served[:40].count("gold") == pytest.approx(30, abs=1)
    await asyncio.gather(*tasks)


class FailingBackend:
    def __init__(self) -> None:
        self.calls = 0

    async def reserve(self, tenant, tokens, requests, limits) -> Reservation:
        self.calls += 1
        raise ConnectionError("redis is down")

    async def adjust(self, tenant, tokens, requests, limits) -> None:
        raise ConnectionError("redis is down")

    async def aclose(self) -> None:
        pass


@virtual
async def test_backend_failure_fails_open_by_default():
    backend = FailingBackend()
    c = AdmissionController(max_inflight_tokens=1000, tenant_tpm=10, backend=backend)
    async with c.admit(tenant="t", cost=500) as ticket:
        assert ticket.remaining_tokens is None  # unknown, not zero
    assert backend.calls == 1
    assert c.metrics.value("admitq_backend_errors_total", operation="reserve") == 1
    assert c.snapshot().inflight_tokens == 0
    await c.aclose()


@virtual
async def test_backend_failure_can_fail_closed():
    c = AdmissionController(tenant_tpm=10, backend=FailingBackend(), on_backend_error="closed")
    with pytest.raises(Shed) as info:
        await c.acquire(tenant="t", cost=1)
    assert (info.value.status, info.value.reason) == (503, "backend_unavailable")
    assert c.metrics.total("admitq_shed_total", reason="backend_unavailable") == 1


@virtual
async def test_backend_is_not_called_for_unlimited_tenants():
    backend = FailingBackend()
    c = AdmissionController(max_inflight_tokens=10, tenant_tpm={"metered": 5}, backend=backend)
    (await c.acquire(tenant="free", cost=5)).release()
    assert backend.calls == 0


class AsyncOnlyBackend:
    """Wraps the memory backend but hides its synchronous fast path (like Redis)."""

    def __init__(self, clock) -> None:
        self.inner = MemoryBackend(clock=clock)
        self.adjustments: list[tuple[str, int, int]] = []
        self.closed = False

    async def reserve(self, tenant, tokens, requests, limits) -> Reservation:
        await asyncio.sleep(0.001)
        return self.inner.reserve_sync(tenant, tokens, requests, limits)

    async def adjust(self, tenant, tokens, requests, limits) -> None:
        await asyncio.sleep(0.001)
        self.adjustments.append((tenant, tokens, requests))
        self.inner.adjust_sync(tenant, tokens, requests, limits)

    async def aclose(self) -> None:
        self.closed = True


@virtual
async def test_async_backend_settles_in_the_background():
    loop = asyncio.get_running_loop()
    backend = AsyncOnlyBackend(loop.time)
    c = AdmissionController(
        max_inflight_requests=1, max_queue_s=0.5, tenant_tpm=1000, backend=backend
    )
    async with c.admit(tenant="t", cost=1000) as ticket:
        ticket.report_usage(prompt_tokens=100, completion_tokens=100)
        with pytest.raises(Shed):  # queue timeout while the first holds the only slot
            await c.acquire(tenant="u", cost=300)
    await c.aclose()  # waits for pending settlements
    assert sorted(backend.adjustments) == [("t", -800, 0), ("u", -300, -1)]
    assert backend.closed


@virtual
async def test_tenant_label_cardinality_is_bounded():
    c = AdmissionController(max_inflight_tokens=10, max_tenant_labels=3)
    for index in range(10):
        (await c.acquire(tenant=f"t{index}", cost=1)).release()
    assert c.metrics.value("admitq_admitted_total", tenant="t0", priority="interactive") == 1
    assert c.metrics.value("admitq_admitted_total", tenant="t2", priority="interactive") == 1
    assert c.metrics.value("admitq_admitted_total", tenant="t3", priority="interactive") == 0
    assert c.metrics.value("admitq_admitted_total", tenant="_other", priority="interactive") == 7
    # Enforcement is still per real tenant; only the label is folded.
    assert c.scheduler.tenant_inflight_tokens("t9") == 0


@virtual
async def test_aclose_sheds_whatever_is_still_queued():
    c = AdmissionController(max_inflight_requests=1, max_queue_s=3600.0, early_shed=False)
    await c.acquire(cost=1)
    waiter = asyncio.create_task(c.acquire(cost=1))
    await settle()
    await c.aclose()
    with pytest.raises(Shed):
        await waiter
    assert c.snapshot().queued_requests == 0
    with pytest.raises(RuntimeError, match="closed"):
        await c.acquire(cost=1)


@virtual
async def test_gauges_track_state():
    c = AdmissionController(max_inflight_tokens=100, max_inflight_requests=5, max_queue_s=60.0)
    assert c.metrics.value("admitq_inflight_token_limit") == 100
    assert c.metrics.value("admitq_inflight_request_limit") == 5
    a = await c.acquire(cost=80)
    waiter = asyncio.create_task(c.acquire(cost=70))
    await settle()
    assert c.metrics.value("admitq_inflight_tokens") == 80
    assert c.metrics.value("admitq_inflight_requests") == 1
    assert c.metrics.value("admitq_queue_depth") == 1
    assert c.metrics.value("admitq_queued_tokens") == 70
    a.release()
    (await waiter).release()
    assert c.metrics.value("admitq_inflight_tokens") == 0
    assert c.metrics.value("admitq_queue_depth") == 0


def test_config_object_or_options_not_both():
    cfg = AdmissionConfig(max_inflight_tokens=10)
    assert AdmissionController(cfg).config is cfg
    with pytest.raises(TypeError, match="not both"):
        AdmissionController(cfg, max_queue_s=1.0)
    with pytest.raises(TypeError):
        AdmissionController(no_such_option=1)


@pytest.mark.parametrize(
    "options",
    [
        {"max_inflight_tokens": 0},
        {"max_inflight_requests": 0},
        {"max_queue_s": -1},
        {"max_queue_depth": -1},
        {"oversize": "explode"},
        {"on_backend_error": "maybe"},
        {"overload_status": 500},
        {"burst_s": 0},
        {"max_tenant_share": 2.0},
        {"retry_after_min": 5.0, "retry_after_max": 1.0},
        {"max_tenant_labels": 0},
        {"tenant_tpm": {"a": 0}},
        {"tenant_rpm": -5},
        {"tenant_weights": {"a": 0}},
        {"priorities": {}},
        {"priorities": [PriorityClass("a", 0), PriorityClass("a", 1)]},
        {"default_priority": "missing"},
    ],
)
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ConfigError):
        AdmissionConfig(**options)


def test_priority_class_validation_and_lookups():
    with pytest.raises(ConfigError):
        PriorityClass("x", -1)
    with pytest.raises(ConfigError):
        PriorityClass("x", 0, aging_s=0)
    with pytest.raises(ConfigError):
        PriorityClass("x", 0, max_queue_s=-1)
    cfg = AdmissionConfig(
        max_queue_s=4.0,
        priorities=[PriorityClass("rt", 0), PriorityClass("bulk", 5, max_queue_s=60.0)],
        default_priority="bulk",
        tenant_tpm={"default": 100, "big": 1000},
        tenant_weights={"default": 2.0, "big": 8.0},
    )
    assert cfg.priority_class(None).name == "bulk"
    assert cfg.priority_class("rt").level == 0
    assert cfg.queue_budget(cfg.priority_class("rt")) == 4.0
    assert cfg.aging_for(cfg.priority_class("rt")) == 2.0
    assert cfg.aging_for(cfg.priority_class("bulk")) == 30.0
    assert cfg.limits_for("big").tpm == 1000
    assert cfg.limits_for("anyone").tpm == 100
    assert cfg.limits_for("anyone").rpm is None
    assert cfg.weight_for("big") == 8.0
    assert cfg.weight_for("anyone") == 2.0
    assert AdmissionConfig(max_queue_s=0.0).aging_for(PriorityClass("x", 1)) == math.inf

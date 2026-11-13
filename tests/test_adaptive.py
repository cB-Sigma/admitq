"""The AIMD limit: unit behaviour and closed-loop convergence in simulation."""

from __future__ import annotations

import asyncio

import pytest

from admitq import AdaptiveLimit, AdmissionController, InferenceAdmission
from admitq.signals import poll_signal, vllm_requests_waiting
from admitq.testing import FakeModelBackend, TenantLoad, call_asgi, run_virtual, simulate_load

from .helpers import tenant_header, virtual


def make(**kwargs) -> AdaptiveLimit:
    defaults = {
        "target": 1.0,
        "min_limit": 10,
        "max_limit": 1000,
        "interval_s": 10.0,
        "min_samples": 3,
    }
    limit = AdaptiveLimit(**{**defaults, **kwargs})
    limit.start(100, now=0.0)
    return limit


def feed(limit: AdaptiveLimit, values, now: float, peak: int | None = None):
    result = None
    if peak is not None:
        limit.note_inflight(peak)
    for value in values:
        result = limit.record(value, now)
    return result


def test_decreases_multiplicatively_when_over_target():
    limit = make(decrease=0.8)
    assert feed(limit, [2.0, 2.0], now=5.0) is None  # window still open
    assert feed(limit, [2.0], now=10.0) == 80
    assert feed(limit, [2.0] * 3, now=20.0) == 64
    assert limit.decreases == 2
    assert limit.limit == 64


def test_increases_additively_only_when_the_limit_is_in_use():
    limit = make(increase=7)
    assert feed(limit, [0.1] * 3, now=10.0, peak=20) is None  # healthy but idle: no change
    assert limit.limit == 100
    assert feed(limit, [0.1] * 3, now=20.0, peak=95) == 107
    assert feed(limit, [0.1] * 3, now=30.0, peak=107) == 114
    assert limit.increases == 2


def test_default_increase_is_one_typical_request():
    limit = make()
    assert limit.increase == 1  # nothing admitted yet
    for _ in range(200):
        limit.note_inflight(100, unit=250.0)  # token dimension: requests cost ~250 tokens
    assert limit.increase == 250
    assert feed(limit, [0.1] * 3, now=10.0) == 350
    requests = make()
    requests.note_inflight(100)  # request dimension: one request is one unit
    assert requests.increase == 1


def test_limits_are_clamped():
    limit = make(min_limit=90, max_limit=103, decrease=0.5, increase=50)
    assert feed(limit, [9.0] * 3, now=10.0) == 90
    assert feed(limit, [9.0] * 3, now=20.0) is None  # already at the floor
    assert feed(limit, [0.0] * 3, now=30.0, peak=90) == 103
    assert feed(limit, [0.0] * 3, now=40.0, peak=103) is None  # already at the ceiling


def test_quantile_decides():
    # One slow sample in ten: p90 is fine, p100 is not.
    samples = [0.1] * 9 + [5.0]
    p90 = make(quantile=0.9, min_samples=10)
    assert feed(p90, samples, now=10.0, peak=100) == 101
    p100 = make(quantile=1.0, min_samples=10)
    assert feed(p100, samples, now=10.0, peak=100) == 85


def test_too_few_samples_extend_the_window():
    limit = make(min_samples=5)
    assert feed(limit, [9.0] * 4, now=50.0) is None
    assert feed(limit, [9.0], now=51.0) == 85


def test_bad_samples_are_ignored_and_start_is_required():
    limit = make()
    assert feed(limit, [float("nan"), float("inf"), -1.0], now=100.0) is None
    assert limit.limit == 100
    fresh = AdaptiveLimit(target=1.0, min_limit=1, max_limit=10)
    with pytest.raises(RuntimeError):
        fresh.record(1.0, 0.0)
    with pytest.raises(RuntimeError):
        _ = fresh.limit


def test_start_clamps_the_configured_limit_and_handles_none():
    high = AdaptiveLimit(target=1.0, min_limit=10, max_limit=50)
    assert high.start(500, now=0.0) == 50
    unset = AdaptiveLimit(target=1.0, min_limit=10, max_limit=50)
    assert unset.start(None, now=0.0) == 50
    explicit = AdaptiveLimit(target=1.0, min_limit=10, max_limit=50, initial=20)
    assert explicit.start(500, now=0.0) == 20


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target": 0},
        {"min_limit": 0},
        {"min_limit": 10, "max_limit": 5},
        {"decrease": 1.0},
        {"decrease": 0.0},
        {"quantile": 0.0},
        {"interval_s": 0},
        {"initial": 5000},
    ],
)
def test_validation(kwargs):
    defaults = {"target": 1.0, "min_limit": 1, "max_limit": 100}
    with pytest.raises(ValueError):  # noqa: PT011
        AdaptiveLimit(**{**defaults, **kwargs})


@virtual
async def test_controller_applies_new_limits_and_wakes_waiters():
    adaptive = AdaptiveLimit(
        target=1.0,
        min_limit=1,
        max_limit=4,
        initial=1,
        dimension="requests",
        signal="reported",
        interval_s=1.0,
        min_samples=1,
        increase=1,
    )
    c = AdmissionController(max_inflight_requests=1, max_queue_s=3600.0, adaptive=adaptive)
    first = await c.acquire(cost=1)
    waiter = asyncio.create_task(c.acquire(cost=1))
    await asyncio.sleep(2.0)
    assert not waiter.done()
    c.report_signal(0.0)  # healthy and fully used: the limit grows to 2 ...
    second = await asyncio.wait_for(waiter, 1.0)  # ... and the waiter gets in
    assert c.snapshot().limit_requests == 2
    assert c.metrics.value("admitq_inflight_request_limit") == 2
    await asyncio.sleep(2.0)
    c.report_signal(50.0)  # congested: back off
    assert c.snapshot().limit_requests == 1
    first.release()
    second.release()
    # report_signal is ignored when the limit listens to time-to-first-byte instead.
    other = AdmissionController(
        max_inflight_requests=4, adaptive=AdaptiveLimit(target=1.0, min_limit=1, max_limit=8)
    )
    other.report_signal(1e9)
    assert other.snapshot().limit_requests == 4


def simulate(initial: int, *, adaptive: bool, duration_s: float = 900.0):
    """An engine that serves 8 requests at once; anything beyond that queues inside it."""
    backend = FakeModelBackend(max_num_seqs=8, total_tps=40, stream_tps=5)
    options = {}
    if adaptive:
        options["adaptive"] = AdaptiveLimit(
            target=0.5, min_limit=1, max_limit=64, dimension="requests", interval_s=10.0
        )
    app = InferenceAdmission(
        backend, max_inflight_requests=initial, max_queue_s=2.0, tenant_key=tenant_header, **options
    )
    history: list[int] = []

    async def main():
        async def sample():
            while True:
                await asyncio.sleep(5.0)
                history.append(app.controller.snapshot().limit_requests)

        sampler = asyncio.create_task(sample())
        report = await simulate_load(
            app, [TenantLoad("t", rps=12, prompt_tokens=100, max_tokens=10)], duration_s=duration_s
        )
        sampler.cancel()
        return report

    report = run_virtual(main())
    return app, backend, report, history


@pytest.mark.parametrize("initial", [1, 4, 32, 64])
def test_aimd_converges_to_engine_capacity_from_any_start(initial):
    app, _backend, report, history = simulate(initial, adaptive=True)
    settled = history[len(history) // 2 :]
    # The engine really fits 8.  AIMD probes just above and backs off: it must end up in
    # a band around 8 whether it started far below or far above.
    assert min(settled) >= 5, (initial, history)
    assert max(settled) <= 12, (initial, history)
    assert 6.5 <= sum(settled) / len(settled) <= 10.5, (initial, history)
    assert app.controller.adaptive.decreases >= 1
    assert app.controller.adaptive.increases >= 1
    # And it still does useful work: close to the engine's ~4 requests/s.
    assert report.overall.ok / report.duration_s > 3.0


def test_adaptive_limit_beats_a_badly_chosen_static_limit():
    _, static_backend, static_report, _ = simulate(64, adaptive=False)
    _, _, adaptive_report, _ = simulate(64, adaptive=True)
    # Static limit 8x too high: dozens of requests pile up inside the engine and time to
    # first byte is dominated by the engine's own queue.
    assert static_backend.peak_waiting >= 40
    assert static_report.overall.ttfb_p50 > 5.0
    # The adaptive limit finds the capacity; the queue moves to admitq, where it is
    # bounded, fair and visible.
    late = [r for r in adaptive_report.records if r.sent_at > 450 and r.outcome == "ok"]
    late_ttfb = sorted(r.ttfb - r.queue_s for r in late)
    assert late_ttfb[len(late_ttfb) // 2] < 0.6
    assert adaptive_report.overall.ttfb_p50 < static_report.overall.ttfb_p50 / 3


def test_adaptive_token_limit_in_simulation():
    backend = FakeModelBackend(max_num_seqs=8, total_tps=40, stream_tps=5)
    adaptive = AdaptiveLimit(target=0.5, min_limit=100, max_limit=40_000, interval_s=10.0)
    app = InferenceAdmission(
        backend,
        max_inflight_tokens=30_000,
        max_queue_s=2.0,
        adaptive=adaptive,
        tenant_key=tenant_header,
    )
    run_virtual(
        simulate_load(
            app, [TenantLoad("t", rps=12, prompt_tokens=100, max_tokens=10)], duration_s=1800
        )
    )
    # Each request is estimated at 114 tokens, so the engine's 8 slots are ~912 tokens.
    assert adaptive.dimension is None  # defaulted to tokens because a token limit is set
    assert adaptive.increase == pytest.approx(114, abs=2)  # one typical request
    assert 600 <= app.controller.snapshot().limit_tokens <= 1500
    assert app.controller.metrics.value("admitq_inflight_token_limit") == adaptive.limit


@virtual
async def test_poll_signal_feeds_engine_queue_depth():
    backend = FakeModelBackend(max_num_seqs=1, stream_tps=10)
    adaptive = AdaptiveLimit(
        target=2.0,
        min_limit=1,
        max_limit=16,
        dimension="requests",
        signal="reported",
        interval_s=5.0,
        min_samples=3,
        increase=1,
    )
    app = InferenceAdmission(backend, max_inflight_requests=16, max_queue_s=1.0, adaptive=adaptive)
    fetches = 0

    async def fetch() -> str:
        nonlocal fetches
        fetches += 1
        if fetches == 3:
            raise ConnectionError("scrape failed")  # skipped, not fatal
        return (await call_asgi(backend, "GET", "/metrics")).body.decode()

    poller = asyncio.create_task(
        poll_signal(app.controller, fetch, parse=vllm_requests_waiting, interval_s=1.0)
    )
    load = simulate_load(app, [TenantLoad("t", rps=4, max_tokens=20)], duration_s=120)
    await load
    poller.cancel()
    assert fetches > 100
    # The engine runs one request at a time; with a target of two waiting, the limit
    # must have come down from 16 to a handful.
    assert app.controller.snapshot().limit_requests <= 5
    assert adaptive.decreases >= 3

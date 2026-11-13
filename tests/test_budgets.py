"""Token-bucket budgets: arithmetic, the in-memory backend and its properties."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from admitq.backends import MemoryBackend, TenantLimits
from admitq.testing import FakeClock


def backend() -> tuple[MemoryBackend, FakeClock]:
    clock = FakeClock()
    return MemoryBackend(clock=clock), clock


def test_bucket_starts_full_and_refills_at_tpm():
    b, clock = backend()
    limits = TenantLimits(tpm=6000)  # 100 tokens/s, capacity 6000
    assert limits.token_rate == 100
    assert limits.token_capacity == 6000
    first = b.reserve_sync("t", 6000, 1, limits)
    assert first.allowed
    assert first.remaining_tokens == 0
    denied = b.reserve_sync("t", 500, 1, limits)
    assert not denied.allowed
    assert denied.reason == "tenant_tokens"
    assert denied.retry_after == pytest.approx(5.0)  # exact: 500 tokens at 100/s
    clock.advance(4.99)
    assert not b.reserve_sync("t", 500, 1, limits).allowed
    clock.advance(0.02)
    assert b.reserve_sync("t", 500, 1, limits).allowed


def test_denied_reservation_debits_nothing():
    b, clock = backend()
    limits = TenantLimits(tpm=600, rpm=60)
    assert b.reserve_sync("t", 600, 1, limits).allowed
    for _ in range(10):
        assert not b.reserve_sync("t", 100, 1, limits).allowed
    clock.advance(10.0)  # 100 tokens refilled; the ten denials above cost nothing
    result = b.reserve_sync("t", 100, 1, limits)
    assert result.allowed
    # 59 left after the first, refilled to the cap of 60, minus this one.  Had the ten
    # denials been debited it would be 58.
    assert result.remaining_requests == pytest.approx(59)


def test_rpm_budget_and_reason():
    b, clock = backend()
    limits = TenantLimits(rpm=120)  # 2/s, capacity 120
    for _ in range(120):
        assert b.reserve_sync("t", 10**9, 1, limits).allowed  # tokens unlimited
    denied = b.reserve_sync("t", 1, 1, limits)
    assert not denied.allowed
    assert denied.reason == "tenant_requests"
    assert denied.retry_after == pytest.approx(0.5)
    assert denied.remaining_tokens is None
    clock.advance(0.5)
    assert b.reserve_sync("t", 1, 1, limits).allowed


def test_both_budgets_report_the_longer_wait():
    b, _ = backend()
    limits = TenantLimits(tpm=60, rpm=60)  # 1 token/s and 1 request/s
    assert b.reserve_sync("t", 60, 60, limits).allowed
    denied = b.reserve_sync("t", 30, 2, limits)
    assert denied.reason == "tenant_tokens"
    assert denied.retry_after == pytest.approx(30.0)


def test_tenants_are_independent():
    b, _ = backend()
    limits = TenantLimits(tpm=100)
    assert b.reserve_sync("a", 100, 1, limits).allowed
    assert not b.reserve_sync("a", 1, 1, limits).allowed
    assert b.reserve_sync("b", 100, 1, limits).allowed


def test_burst_s_shrinks_the_bucket():
    b, clock = backend()
    limits = TenantLimits(tpm=6000, burst_s=10)  # 100/s, capacity 1000
    assert limits.token_capacity == 1000
    assert b.reserve_sync("t", 1000, 1, limits).allowed
    assert not b.reserve_sync("t", 1, 1, limits).allowed
    clock.advance(3600)
    assert b.reserve_sync("t", 1000, 1, limits).remaining_tokens == 0  # capped at 1000


def test_oversized_request_is_admitted_on_a_full_bucket_and_leaves_debt():
    b, clock = backend()
    limits = TenantLimits(tpm=600)  # 10/s, capacity 600
    big = b.reserve_sync("t", 1500, 1, limits)
    assert big.allowed  # needs min(cost, capacity) = a full bucket
    assert big.remaining_tokens == 0
    denied = b.reserve_sync("t", 10, 1, limits)
    assert denied.retry_after == pytest.approx((900 + 10) / 10)  # pay the debt first
    clock.advance(91.0)
    assert b.reserve_sync("t", 10, 1, limits).allowed


def test_adjust_charges_and_refunds():
    b, clock = backend()
    limits = TenantLimits(tpm=600, rpm=60)
    assert b.reserve_sync("t", 300, 1, limits).allowed
    b.adjust_sync("t", -200, 0, limits)  # really cost 100: refund 200
    assert b.reserve_sync("t", 500, 1, limits).allowed
    b.adjust_sync("t", 250, 0, limits)  # under-estimated: charge 250 more -> debt
    denied = b.reserve_sync("t", 1, 1, limits)
    assert denied.retry_after == pytest.approx(25.1)
    b.adjust_sync("t", -10_000, -100, limits)  # refunds never overfill the bucket
    clock.advance(0.0)
    full = b.reserve_sync("t", 600, 1, limits)
    assert full.allowed
    assert full.remaining_tokens == 0
    assert full.remaining_requests == 59


def test_debt_is_capped_at_one_bucket():
    b, clock = backend()
    limits = TenantLimits(tpm=600)
    b.reserve_sync("t", 600, 1, limits)
    b.adjust_sync("t", 10**9, 0, limits)
    clock.advance(121.0)  # -600 -> 0 takes 60 s, then 10 tokens need 1 more
    assert b.reserve_sync("t", 600, 1, limits).allowed


def test_unlimited_dimensions():
    limits = TenantLimits()
    assert limits.unlimited
    b, _ = backend()
    result = b.reserve_sync("t", 10**12, 10**6, limits)
    assert result.allowed
    assert result.remaining_tokens is None
    assert result.remaining_requests is None


def test_tiny_rpm_still_admits_one_request():
    b, clock = backend()
    limits = TenantLimits(rpm=1, burst_s=1)  # capacity would be 1/60 of a request
    assert limits.request_capacity == 1.0
    assert b.reserve_sync("t", 0, 1, limits).allowed
    assert not b.reserve_sync("t", 0, 1, limits).allowed
    clock.advance(60.0)
    assert b.reserve_sync("t", 0, 1, limits).allowed


async def test_async_interface_matches_sync():
    b, _ = backend()
    limits = TenantLimits(tpm=100)
    assert (await b.reserve("t", 100, 1, limits)).allowed
    await b.adjust("t", -100, 0, limits)
    assert (await b.reserve("t", 100, 1, limits)).allowed
    await b.aclose()


def test_idle_buckets_are_swept():
    b, clock = backend()
    limits = TenantLimits(tpm=600)
    for index in range(10_000):
        b.reserve_sync(f"t{index}", 1, 1, limits)
    clock.advance(1000.0)
    b.reserve_sync("fresh", 1, 1, limits)
    assert len(b._buckets) == 1


@given(
    tpm=st.integers(60, 600_000),
    burst_s=st.sampled_from([1.0, 10.0, 60.0]),
    events=st.lists(st.tuples(st.floats(0.0, 30.0), st.integers(0, 5000)), max_size=80),
)
def test_admitted_tokens_never_exceed_bucket_plus_refill(tpm, burst_s, events):
    """Token-bucket conformance: over any run, the tokens admitted are at most one full
    bucket plus what was refilled (plus one oversized request's excess, which is the
    documented debt rule)."""
    b, clock = backend()
    limits = TenantLimits(tpm=tpm, burst_s=burst_s)
    admitted = 0
    largest = 0
    for gap, cost in events:
        clock.advance(gap)
        result = b.reserve_sync("t", cost, 1, limits)
        assert result.remaining_tokens is not None
        assert 0 <= result.remaining_tokens <= limits.token_capacity + 1e-6
        if result.allowed:
            admitted += cost
            largest = max(largest, cost)
        else:
            assert result.retry_after > 0
            # The promise in Retry-After is exact: waiting that long is enough.
            probe_clock = FakeClock(clock.now)
            probe = MemoryBackend(clock=probe_clock)
            probe._buckets = {k: list(v) for k, v in b._buckets.items()}
            probe_clock.advance(result.retry_after * (1 + 1e-9) + 1e-9)
            assert probe.reserve_sync("t", cost, 1, limits).allowed
    excess = max(0.0, largest - limits.token_capacity)
    assert admitted <= limits.token_capacity + limits.token_rate * clock.now + excess + 1e-6


@given(
    estimates=st.lists(st.integers(1, 2000), min_size=1, max_size=40),
    actual_ratio=st.floats(0.0, 2.0),
)
def test_post_hoc_correction_converges_to_real_usage(estimates, actual_ratio):
    """After correction the bucket is where it would be had the real cost been charged
    up front (as long as no clamp is hit)."""
    limits = TenantLimits(tpm=10_000_000)
    corrected, _ = backend()
    oracle, _ = backend()
    for estimate in estimates:
        actual = int(estimate * actual_ratio)
        corrected.reserve_sync("t", estimate, 1, limits)
        corrected.adjust_sync("t", actual - estimate, 0, limits)
        oracle.reserve_sync("t", actual, 1, limits)
    assert corrected._buckets["t"][0] == pytest.approx(oracle._buckets["t"][0])

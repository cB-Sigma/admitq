"""The Redis budget backend against a real ``redis-server``.

The server is taken from ``ADMITQ_REDIS_URL`` if set (CI uses a service container),
otherwise a private ``redis-server`` is started on a free port if the binary is on PATH.
With neither, these tests are skipped.  Redis keeps its own clock, so these tests run in
real time with generous tolerances.
"""

import asyncio
import os
import shutil
import socket
import subprocess
import time
import uuid

import pytest

redis_asyncio = pytest.importorskip("redis.asyncio")

from admitq import AdmissionController, InferenceAdmission, Shed, TenantLimits  # noqa: E402
from admitq.backends.redis import RedisBackend  # noqa: E402
from admitq.testing import FakeModelBackend, call_asgi  # noqa: E402

pytestmark = pytest.mark.redis


@pytest.fixture(scope="module")
def redis_url(tmp_path_factory):
    url = os.environ.get("ADMITQ_REDIS_URL")
    if url:
        yield url
        return
    binary = shutil.which("redis-server")
    if binary is None:
        pytest.skip("no ADMITQ_REDIS_URL and no redis-server binary on PATH")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    workdir = tmp_path_factory.mktemp("redis")
    base = [binary, "--port", str(port), "--bind", "127.0.0.1", "--save", "", "--appendonly", "no"]
    base += ["--dir", str(workdir)]
    process = None
    # DEBUG SLEEP (used to simulate a stalled server) needs an opt-in on Redis >= 7; older
    # servers do not know the option, so fall back to starting without it.
    for extra in (["--enable-debug-command", "local"], []):
        process = subprocess.Popen(
            base + extra, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        deadline = time.monotonic() + 10
        up = False
        while time.monotonic() < deadline and process.poll() is None:
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    up = True
                    break
            time.sleep(0.05)
        if up:
            break
        process.kill()
        process.wait(timeout=10)
        process = None
    if process is None:
        pytest.skip("redis-server did not start")
    try:
        yield f"redis://127.0.0.1:{port}/0"
    finally:
        process.terminate()
        process.wait(timeout=10)


@pytest.fixture
async def client(redis_url):
    connection = redis_asyncio.Redis.from_url(redis_url)
    yield connection
    await connection.aclose()


@pytest.fixture
def prefix():
    return f"admitq-test-{uuid.uuid4().hex[:8]}"


async def test_reserve_debit_deny_and_exact_retry_after(client, prefix):
    backend = RedisBackend(client, prefix=prefix)
    limits = TenantLimits(tpm=6000, rpm=600)  # 100 tokens/s, 10 requests/s
    first = await backend.reserve("t", 5900, 1, limits)
    assert first.allowed
    assert first.remaining_tokens == pytest.approx(100, abs=5)
    assert first.remaining_requests == pytest.approx(599, abs=1)
    denied = await backend.reserve("t", 150, 1, limits)
    assert not denied.allowed
    assert denied.reason == "tenant_tokens"
    assert 0.3 < denied.retry_after <= 0.5
    # A denial debits nothing, and the promised wait is enough.
    await asyncio.sleep(denied.retry_after + 0.02)
    assert (await backend.reserve("t", 150, 1, limits)).allowed
    await backend.aclose()


async def test_request_budget_reason(client, prefix):
    backend = RedisBackend(client, prefix=prefix)
    limits = TenantLimits(rpm=60, burst_s=2)  # 1/s, capacity 2
    assert (await backend.reserve("t", 10**9, 1, limits)).allowed
    assert (await backend.reserve("t", 10**9, 1, limits)).allowed
    denied = await backend.reserve("t", 1, 1, limits)
    assert not denied.allowed
    assert denied.reason == "tenant_requests"
    assert denied.remaining_tokens is None
    assert 0.5 < denied.retry_after <= 1.0


async def test_adjust_refunds_charges_and_caps(client, prefix):
    backend = RedisBackend(client, prefix=prefix)
    limits = TenantLimits(tpm=60_000)  # capacity 60 000, 1000/s
    assert (await backend.reserve("t", 50_000, 1, limits)).allowed
    await backend.adjust("t", -45_000, 0, limits)  # really cost 5 000
    after = await backend.reserve("t", 0, 1, limits)
    assert after.remaining_tokens == pytest.approx(55_000, abs=300)
    await backend.adjust("t", 10**9, 0, limits)  # debt is capped at one bucket
    denied = await backend.reserve("t", 1, 1, limits)
    assert 55.0 < denied.retry_after <= 60.1
    await backend.adjust("t", -(10**9), 0, limits)  # refunds never overfill
    assert (await backend.reserve("t", 0, 1, limits)).remaining_tokens == pytest.approx(
        60_000, abs=1
    )


async def test_oversized_request_runs_on_a_full_bucket(client, prefix):
    backend = RedisBackend(client, prefix=prefix)
    limits = TenantLimits(tpm=600)
    big = await backend.reserve("t", 5000, 1, limits)
    assert big.allowed
    assert big.remaining_tokens == 0
    assert not (await backend.reserve("t", 1, 1, limits)).allowed


async def test_keys_are_namespaced_hash_tagged_and_expire(client, prefix):
    backend = RedisBackend(client, prefix=prefix)
    limits = TenantLimits(tpm=600, burst_s=30)
    await backend.reserve("acme", 1, 1, limits)
    key = f"{prefix}:{{acme}}"
    assert await client.exists(key) == 1
    assert set(await client.hkeys(key)) == {b"tok", b"req", b"ts"}
    ttl = await client.pttl(key)
    assert 50_000 < ttl <= 60_000  # twice the burst window
    assert await client.exists(f"{prefix}:{{other}}") == 0


async def test_concurrent_workers_cannot_overspend(redis_url, prefix):
    """Atomicity: 400 concurrent reservations from four separate connections ("workers")
    against a bucket of 1000 tokens.  Exactly as many succeed as the bucket (plus refill
    during the test) can pay for."""
    clients = [redis_asyncio.Redis.from_url(redis_url) for _ in range(4)]
    backends = [RedisBackend(c, prefix=prefix, timeout=5.0) for c in clients]
    limits = TenantLimits(tpm=1000)  # 16.7 tokens/s refill
    started = time.monotonic()
    results = await asyncio.gather(
        *(backends[i % 4].reserve("shared", 10, 1, limits) for i in range(400))
    )
    elapsed = time.monotonic() - started
    admitted = sum(r.allowed for r in results)
    assert admitted >= 100
    assert admitted * 10 <= 1000 + limits.token_rate * (elapsed + 0.05)
    for connection in clients:
        await connection.aclose()


async def test_two_controllers_share_one_budget(redis_url, prefix):
    one = AdmissionController(
        tenant_tpm=1000, backend=RedisBackend.from_url(redis_url, prefix=prefix, timeout=2.0)
    )
    two = AdmissionController(
        tenant_tpm=1000, backend=RedisBackend.from_url(redis_url, prefix=prefix, timeout=2.0)
    )
    ticket = await one.acquire(tenant="t", cost=900)
    with pytest.raises(Shed) as info:
        await two.acquire(tenant="t", cost=900)  # the other "worker" sees the spend
    assert info.value.reason == "tenant_tokens"
    assert info.value.status == 429
    # Correction through one worker is visible to the other.
    ticket.report_usage(prompt_tokens=50, completion_tokens=50)
    ticket.release()
    await one.aclose()  # flushes the pending adjustment
    other = await two.acquire(tenant="t", cost=850)
    assert other.remaining_tokens == pytest.approx(50, abs=20)
    other.release()
    await two.aclose()


async def test_in_flight_limits_are_per_process_even_with_redis(redis_url, prefix):
    """Documented behaviour, pinned by a test: Redis shares budgets, not concurrency."""
    workers = [
        AdmissionController(
            max_inflight_requests=1,
            max_queue_s=0.0,
            tenant_tpm=10**9,
            backend=RedisBackend.from_url(redis_url, prefix=prefix, timeout=2.0),
        )
        for _ in range(2)
    ]
    tickets = [await worker.acquire(tenant="t", cost=1) for worker in workers]
    assert len(tickets) == 2  # one in flight per worker, two in total
    with pytest.raises(Shed):
        await workers[0].acquire(tenant="t", cost=1)
    for ticket in tickets:
        ticket.release()
    for worker in workers:
        await worker.aclose()


async def test_middleware_end_to_end_with_redis(redis_url, prefix):
    backend = RedisBackend.from_url(redis_url, prefix=prefix, timeout=2.0)
    engine = FakeModelBackend(stream_tps=2000, total_tps=20_000)
    app = InferenceAdmission(engine, tenant_tpm=2000, backend=backend, max_inflight_tokens=10_000)
    body = {
        "messages": [{"role": "user", "content": "x" * 400}],
        "max_tokens": 1500,
        "x_fake_output_tokens": 20,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    first = await call_asgi(app, "POST", "/v1/chat/completions", json_body=body)
    assert first.status == 200
    assert first.headers["x-ratelimit-limit-tokens"] == "2000"
    await asyncio.sleep(0.1)  # the correction is applied in the background
    # 1604 were reserved, 120 used: the second request fits only because of the refund.
    second = await call_asgi(app, "POST", "/v1/chat/completions", json_body=body)
    assert second.status == 200
    await asyncio.sleep(0.1)
    body["x_fake_output_tokens"] = 1500
    third = await call_asgi(app, "POST", "/v1/chat/completions", json_body=body)
    assert third.status == 200
    shed = await call_asgi(app, "POST", "/v1/chat/completions", json_body=body)
    assert shed.status == 429
    assert shed.headers["x-admitq-shed-reason"] == "tenant_tokens"
    await app.controller.aclose()


async def test_unreachable_redis_fails_open_or_closed(prefix):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        dead_port = sock.getsockname()[1]
    url = f"redis://127.0.0.1:{dead_port}/0"
    open_controller = AdmissionController(
        tenant_tpm=10, max_inflight_tokens=100, backend=RedisBackend.from_url(url, prefix=prefix)
    )
    ticket = await open_controller.acquire(tenant="t", cost=50)  # admitted despite the outage
    assert ticket.remaining_tokens is None
    ticket.release()
    assert open_controller.metrics.value("admitq_backend_errors_total", operation="reserve") == 1
    assert open_controller.snapshot().inflight_tokens == 0
    await open_controller.aclose()

    closed_controller = AdmissionController(
        tenant_tpm=10, backend=RedisBackend.from_url(url, prefix=prefix), on_backend_error="closed"
    )
    with pytest.raises(Shed) as info:
        await closed_controller.acquire(tenant="t", cost=1)
    assert info.value.reason == "backend_unavailable"
    await closed_controller.aclose()


async def test_slow_redis_is_bounded_by_the_timeout(redis_url, prefix):
    connection = redis_asyncio.Redis.from_url(redis_url)
    backend = RedisBackend(connection, prefix=prefix, timeout=0.05)
    blocker = redis_asyncio.Redis.from_url(redis_url)
    # Stall the server for 400 ms; the backend must give up after 50 ms, not hang.
    stall = asyncio.create_task(blocker.execute_command("DEBUG", "SLEEP", "0.4"))
    await asyncio.sleep(0.05)
    started = time.monotonic()
    try:
        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            await backend.reserve("t", 1, 1, TenantLimits(tpm=100))
        assert time.monotonic() - started < 0.3
    finally:
        try:
            await stall
        except Exception:
            pytest.skip("DEBUG SLEEP is not allowed on this server")
        await connection.aclose()
        await blocker.aclose()

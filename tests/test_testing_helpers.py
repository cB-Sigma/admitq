"""admitq.testing is public API: test it like the rest."""

from __future__ import annotations

import asyncio
import math
import time

import pytest

from admitq import InferenceAdmission
from admitq.testing import (
    FakeClock,
    FakeModelBackend,
    TenantLoad,
    VirtualTimeLoop,
    call_asgi,
    percentile,
    run_virtual,
    simulate_load,
)

from .helpers import tenant_header, virtual

CHAT = "/v1/chat/completions"


def test_fake_clock():
    clock = FakeClock(10.0)
    assert clock() == 10.0
    assert clock.advance(2.5) == 12.5
    with pytest.raises(ValueError, match="forward"):
        clock.advance(-1)


def test_virtual_time_is_instant_and_exact():
    async def main() -> float:
        loop = asyncio.get_running_loop()
        assert isinstance(loop, VirtualTimeLoop)
        await asyncio.sleep(3600)
        await asyncio.gather(asyncio.sleep(10), asyncio.sleep(20))
        return loop.time()

    started = time.perf_counter()
    assert run_virtual(main()) == pytest.approx(3620.0)
    assert time.perf_counter() - started < 1.0


def test_virtual_loop_detects_a_deadlock_instead_of_hanging():
    async def main() -> None:
        await asyncio.get_running_loop().create_future()  # nobody will ever resolve this

    with pytest.raises(RuntimeError, match="never happen"):
        run_virtual(main())


def test_run_virtual_cleans_up_leftover_tasks():
    cancelled = []

    async def background() -> None:
        try:
            await asyncio.sleep(10**9)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    async def main() -> int:
        asyncio.get_running_loop().create_task(background())
        await asyncio.sleep(1)
        return 7

    assert run_virtual(main()) == 7
    assert cancelled == [True]


def test_percentile():
    assert percentile([], 50) != percentile([], 50)  # nan
    assert percentile([5], 99) == 5
    values = list(range(1, 101))
    assert percentile(values, 50) == 50
    assert percentile(values, 99) == 99
    assert percentile(values, 100) == 100
    assert percentile(values, 0) == 1


@virtual
async def test_fake_backend_timing_model():
    backend = FakeModelBackend(max_num_seqs=2, total_tps=40, stream_tps=20, prefill_tps=1000)
    body = {"messages": [{"role": "user", "content": "x" * 400}], "max_tokens": 20, "stream": True}
    # Alone: 0.1 s prefill, then 20 tokens at 20/s.
    alone = await call_asgi(backend, "POST", CHAT, json_body=body)
    assert alone.ttfb == pytest.approx(0.15)
    assert alone.duration == pytest.approx(1.1)
    # Four at once on two slots: two run (20 tok/s each), two wait a full generation.
    results = await asyncio.gather(
        *(call_asgi(backend, "POST", CHAT, json_body=body) for _ in range(4))
    )
    durations = sorted(r.duration for r in results)
    assert durations == pytest.approx([1.1, 1.1, 2.2, 2.2])
    assert backend.peak_waiting == 2
    assert backend.peak_running == 2
    assert backend.completed == 5
    assert backend.generated_tokens == 100
    assert backend.prompt_tokens == 500


@virtual
async def test_fake_backend_slows_down_as_the_batch_fills():
    backend = FakeModelBackend(max_num_seqs=8, total_tps=40, stream_tps=20, prefill_tps=1e9)
    body = {"messages": [], "max_tokens": 20}
    results = await asyncio.gather(
        *(call_asgi(backend, "POST", CHAT, json_body=body) for _ in range(8))
    )
    # Eight sequences share 40 tokens/s: 5 tokens/s each instead of 20.
    assert all(r.duration == pytest.approx(4.0) for r in results)
    assert results[0].json()["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 20,
        "total_tokens": 20,
    }


@virtual
async def test_fake_backend_routes_and_options():
    backend = FakeModelBackend(max_num_seqs=1, max_waiting=0, stream_tps=10, report_usage=False)
    assert (await call_asgi(backend, "GET", "/health")).json() == {"status": "ok"}
    assert (await call_asgi(backend, "GET", "/nope")).status == 404
    assert (await call_asgi(backend, "POST", CHAT, body=b"not json")).status == 400
    completion = await call_asgi(
        backend, "POST", "/v1/completions", json_body={"prompt": "abcd" * 10, "max_tokens": 3}
    )
    assert completion.json()["choices"][0]["text"] == "tok tok tok "
    assert "usage" not in completion.json()
    stream = await call_asgi(
        backend,
        "POST",
        "/v1/completions",
        json_body={
            "prompt": "x",
            "max_tokens": 2,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert [event["choices"][0]["text"] for event in stream.sse_events()] == ["tok ", "tok "]
    embeddings = await call_asgi(
        backend, "POST", "/v1/embeddings", json_body={"input": ["abcd", "efgh"]}
    )
    assert len(embeddings.json()["data"]) == 2
    assert embeddings.json()["usage"]["prompt_tokens"] == 2
    # A bounded engine queue (max_waiting=0) refuses instead of queueing.
    slow = asyncio.create_task(call_asgi(backend, "POST", CHAT, json_body={"max_tokens": 20}))
    await asyncio.sleep(0.5)
    metrics = (await call_asgi(backend, "GET", "/metrics")).body.decode()
    assert 'vllm:num_requests_running{model_name="fake-model"} 1' in metrics
    assert (await call_asgi(backend, "POST", CHAT, json_body={"max_tokens": 1})).status == 503
    assert backend.rejected == 1
    await slow


@virtual
async def test_fake_backend_aborts_waiting_requests_on_disconnect():
    backend = FakeModelBackend(max_num_seqs=1, stream_tps=10)
    holder = asyncio.create_task(call_asgi(backend, "POST", CHAT, json_body={"max_tokens": 50}))
    await asyncio.sleep(0.1)
    waiting = await call_asgi(
        backend, "POST", CHAT, json_body={"max_tokens": 50}, disconnect_after_s=1.0
    )
    assert waiting.disconnected
    assert waiting.duration == pytest.approx(1.0)
    assert (backend.waiting, backend.aborted) == (0, 1)
    await holder
    assert backend.running == 0


@virtual
async def test_call_asgi_shapes():
    seen = {}

    async def echo(scope, receive, send):
        seen["scope"] = scope
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 201, "headers": [(b"X-Echo", b"yes")]})
        await send({"type": "http.response.body", "body": body[:3], "more_body": True})
        await send({"type": "http.response.body", "body": body[3:]})

    response = await call_asgi(
        echo, "post", "/p?q=1", body=[b"ab", b"cd", b"ef"], headers={"X-A": "1"}
    )
    assert (response.status, response.body, response.chunks) == (201, b"abcdef", 2)
    assert response.headers == {"x-echo": "yes"}
    assert response.completed
    assert not response.disconnected
    scope = seen["scope"]
    assert (scope["method"], scope["path"], scope["query_string"]) == ("POST", "/p", b"q=1")
    assert (b"x-a", b"1") in scope["headers"]
    assert not any(name == b"content-length" for name, _ in scope["headers"])  # chunked upload
    json_response = await call_asgi(echo, "POST", "/", json_body={"a": 1})
    assert json_response.json() == {"a": 1}
    assert (b"content-type", b"application/json") in seen["scope"]["headers"]
    dropped = await call_asgi(echo, "POST", "/", body=b"abcdef", keep_body=False)
    assert (dropped.body, dropped.chunks) == (b"", 2)


def run_load(app, **kwargs):
    defaults = {"duration_s": 60, "seed": 3}
    tenants = [
        TenantLoad("quiet", rps=1, max_tokens=20),
        TenantLoad("noisy", rps=20, max_tokens=20),
    ]
    return run_virtual(simulate_load(app, tenants, **{**defaults, **kwargs}))


def test_simulate_load_is_deterministic_and_seeded():
    def build():
        return InferenceAdmission(
            FakeModelBackend(max_num_seqs=4, total_tps=80, stream_tps=20),
            max_inflight_requests=4,
            max_queue_s=1.0,
            tenant_key=tenant_header,
        )

    one, two, other = run_load(build()), run_load(build()), run_load(build(), seed=4)
    assert one.records == two.records
    assert one.records != other.records
    assert one.tenants == ["quiet", "noisy"]
    assert one.overall.sent == one.tenant("quiet").sent + one.tenant("noisy").sent
    assert one.tenant("noisy").sent == pytest.approx(1200, rel=0.1)  # Poisson, 20 rps x 60 s


def test_simulate_load_reports_shedding_fairness_and_goodput():
    app = InferenceAdmission(
        FakeModelBackend(max_num_seqs=4, total_tps=80, stream_tps=20),
        max_inflight_requests=4,
        max_queue_s=1.0,
        tenant_key=tenant_header,
    )
    report = run_load(app, slo_s=5.0)
    quiet, noisy = report.tenant("quiet"), report.tenant("noisy")
    assert noisy.shed_rate > 0.7
    assert quiet.shed_rate < 0.15  # fair queuing protects the quiet tenant
    assert quiet.goodput_fraction > 0.85
    assert quiet.ok + quiet.shed + quiet.timeouts + quiet.errors == quiet.sent
    assert quiet.latency_p50 <= quiet.latency_p99
    assert noisy.queue_p99 <= 1.0 + 1e-6  # the queue bound, as seen by clients
    assert report.overall.goodput_rps == pytest.approx(report.overall.good / 60)
    assert report.overall.goodput_tps > 0
    shed = [r for r in report.records if r.outcome == "shed"]
    assert {r.status for r in shed} == {503}
    assert {r.shed_reason for r in shed} <= {"queue_timeout", "predicted_wait", "queue_full"}
    table = report.format()
    assert "quiet" in table
    assert "noisy" in table
    assert "all" in table


def test_simulate_load_without_protection_shows_the_overload():
    backend = FakeModelBackend(max_num_seqs=4, total_tps=80, stream_tps=20)
    report = run_load(backend, slo_s=5.0, client_timeout_s=30.0)
    quiet = report.tenant("quiet")
    assert quiet.shed == 0
    assert quiet.timeouts > quiet.ok  # most of the quiet tenant's requests time out
    assert quiet.goodput_fraction < 0.2
    assert backend.peak_waiting > 100
    assert backend.aborted == report.overall.timeouts


def test_simulate_load_windows_priorities_and_timeout_header():
    seen: list[dict[str, str]] = []
    backend = FakeModelBackend()

    async def spy(scope, receive, send):
        seen.append({k.decode(): v.decode() for k, v in scope["headers"]})
        await backend(scope, receive, send)

    tenants = [
        TenantLoad(
            "burst", rps=10, max_tokens=5, start_s=10, stop_s=20, priority="batch", stream=False
        ),
        TenantLoad("silent", rps=0),
    ]
    report = run_virtual(
        simulate_load(spy, tenants, duration_s=60, client_timeout_s=7.5, send_timeout_header=True)
    )
    burst = report.tenant("burst")
    assert burst.sent == pytest.approx(100, rel=0.35)
    assert all(10 <= r.sent_at < 20 for r in report.records)
    assert report.tenant("silent").sent == 0
    assert math.isnan(report.tenant("silent").latency_p50)
    assert all(h["x-priority"] == "batch" and h["x-request-timeout"] == "7.5" for h in seen)
    assert burst.good_tokens > 0  # non-streaming usage is counted too

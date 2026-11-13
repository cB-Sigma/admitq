"""The ASGI middleware against in-process apps, on a virtual-time event loop."""

from __future__ import annotations

import asyncio
import json

import pytest

from admitq import AdmissionController, InferenceAdmission, RouteMatcher, header_tenant
from admitq.testing import FakeModelBackend, call_asgi

from .helpers import settle, tenant_header, virtual

CHAT = "/v1/chat/completions"


def chat(prompt_tokens: int = 100, max_tokens: int = 20, **extra):
    return {
        "model": "m",
        "messages": [{"role": "user", "content": "x" * (4 * prompt_tokens)}],
        "max_tokens": max_tokens,
        **extra,
    }


class Recorder:
    """A downstream app that records exactly what it was given, then answers 200."""

    def __init__(self, *, read_body: bool = True) -> None:
        self.scopes: list[dict] = []
        self.messages: list[dict] = []
        self.read_body = read_body
        self.calls = 0

    async def __call__(self, scope, receive, send):
        self.calls += 1
        self.scopes.append(scope)
        if self.read_body:
            while True:
                message = await receive()
                self.messages.append(message)
                if message["type"] != "http.request" or not message.get("more_body"):
                    break
        await send({"type": "http.response.start", "status": 200, "headers": [(b"x-app", b"1")]})
        await send({"type": "http.response.body", "body": b"ok"})

    @property
    def body(self) -> bytes:
        return b"".join(m.get("body", b"") for m in self.messages if m["type"] == "http.request")


def idle(app: InferenceAdmission) -> bool:
    snap = app.controller.snapshot()
    return (snap.inflight_tokens, snap.inflight_requests, snap.queued_requests) == (0, 0, 0)


# ------------------------------------------------------------------ pass-through


@virtual
async def test_non_inference_routes_are_untouched():
    seen = {}

    async def downstream(scope, receive, send):
        seen["scope"], seen["receive"], seen["send"] = scope, receive, send
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    app = InferenceAdmission(downstream, max_inflight_requests=1, max_queue_s=0)
    scope = {"type": "http", "method": "POST", "path": "/v1/models", "headers": []}

    async def receive():
        raise AssertionError("the middleware must not read this body")

    async def send(message):
        pass

    await app(scope, receive, send)
    # The very same objects: not wrapped, not copied, not read.
    assert seen == {"scope": scope, "receive": receive, "send": send}
    for path in ("/health", "/v1/chat/completions/extra", "/docs"):
        assert (await call_asgi(app, "POST", path)).status == 204
    assert (await call_asgi(app, "GET", CHAT)).status == 204  # GET is not an inference call
    assert app.controller.metrics.total("admitq_admitted_total") == 0


@virtual
async def test_non_http_scopes_are_passed_through():
    seen = []

    async def downstream(scope, receive, send):
        seen.append(scope["type"])

    app = InferenceAdmission(downstream)
    await app({"type": "lifespan"}, None, None)
    await app({"type": "websocket", "path": CHAT}, None, None)
    assert seen == ["lifespan", "websocket"]


# ------------------------------------------------------------------ body handling


@virtual
async def test_body_is_replayed_byte_for_byte():
    downstream = Recorder()
    app = InferenceAdmission(downstream, max_inflight_tokens=10_000)
    raw = json.dumps(chat(), indent=3).encode() + b"  \n"  # unusual formatting survives
    response = await call_asgi(app, "POST", CHAT, body=raw, headers={"x-custom": "kept"})
    assert response.status == 200
    assert response.body == b"ok"
    assert response.headers["x-app"] == "1"
    assert downstream.body == raw
    assert dict(downstream.scopes[0]["headers"])[b"x-custom"] == b"kept"
    assert dict(downstream.scopes[0]["headers"])[b"content-length"] == str(len(raw)).encode()
    assert idle(app)


@virtual
async def test_chunked_upload_is_reassembled():
    downstream = Recorder()
    app = InferenceAdmission(downstream, max_inflight_tokens=10_000)
    raw = json.dumps(chat(prompt_tokens=500)).encode()
    pieces = [raw[i : i + 97] for i in range(0, len(raw), 97)]
    response = await call_asgi(app, "POST", CHAT, body=pieces)
    assert response.status == 200
    assert downstream.body == raw
    assert len([m for m in downstream.messages if m["type"] == "http.request"]) == 1
    assert int(response.headers["x-admitq-estimated-tokens"]) == 500 + 4 + 20


@virtual
async def test_large_body_with_content_length_is_streamed_not_buffered():
    downstream = Recorder()
    app = InferenceAdmission(downstream, max_inflight_tokens=10**9, max_body_bytes=1024)
    pieces = [b"a" * 1000 for _ in range(50)]
    response = await call_asgi(
        app, "POST", CHAT, body=pieces, headers={"content-length": str(50_000)}
    )
    assert response.status == 200
    requests = [m for m in downstream.messages if m["type"] == "http.request"]
    # An empty replay message, then the client's own 50 chunks one by one.
    assert [len(m["body"]) for m in requests] == [0] + [1000] * 50
    assert downstream.body == b"".join(pieces)
    # Costed from its size: 50 000 bytes / 4 + the default 512 output tokens.
    assert int(response.headers["x-admitq-estimated-tokens"]) == 12_500 + 512
    assert idle(app)


@virtual
async def test_large_chunked_body_stops_buffering_at_the_cap():
    downstream = Recorder()
    app = InferenceAdmission(downstream, max_inflight_tokens=10**9, max_body_bytes=2500)
    pieces = [b"b" * 1000 for _ in range(20)]
    response = await call_asgi(app, "POST", CHAT, body=pieces)  # no content-length
    assert response.status == 200
    requests = [m for m in downstream.messages if m["type"] == "http.request"]
    assert [len(m["body"]) for m in requests] == [3000] + [1000] * 17
    assert downstream.body == b"".join(pieces)
    assert int(response.headers["x-admitq-estimated-tokens"]) == 750 + 512


@virtual
async def test_non_json_empty_and_non_object_bodies_reach_the_app():
    for raw in (b"this is not json", b"", b"[1, 2, 3]", b'"just a string"', b"\xff\xfe\x00", b"{"):
        downstream = Recorder()
        app = InferenceAdmission(downstream, max_inflight_tokens=10_000)
        response = await call_asgi(app, "POST", CHAT, body=raw)
        assert response.status == 200  # the app decides what to make of it, not admitq
        assert downstream.body == raw
        assert idle(app)


@virtual
async def test_deeply_nested_json_does_not_crash():
    downstream = Recorder()
    app = InferenceAdmission(downstream, max_inflight_tokens=10**9)
    raw = b"[" * 100_000 + b"]" * 100_000
    response = await call_asgi(app, "POST", CHAT, body=raw)
    assert response.status == 200
    assert downstream.body == raw


@virtual
async def test_client_disconnect_during_upload_never_reaches_the_app():
    downstream = Recorder()
    app = InferenceAdmission(downstream, max_inflight_tokens=10_000)
    messages = [
        {"type": "http.request", "body": b'{"mess', "more_body": True},
        {"type": "http.disconnect"},
    ]

    async def receive():
        return messages.pop(0)

    sent = []

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": CHAT, "headers": []}
    await app(scope, receive, send)
    assert downstream.calls == 0
    assert sent == []
    assert idle(app)


# ------------------------------------------------------------------ shedding


@virtual
async def test_tenant_budget_shed_is_a_429_with_openai_error_body():
    backend = FakeModelBackend()
    app = InferenceAdmission(backend, tenant_tpm=1000, tenant_key=tenant_header)
    ok = await call_asgi(app, "POST", CHAT, json_body=chat(800, 100), headers={"x-tenant-id": "a"})
    assert ok.status == 200
    assert ok.headers["x-ratelimit-limit-tokens"] == "1000"
    assert ok.headers["x-ratelimit-remaining-tokens"] == "96"  # 1000 - (800 + 4 + 100)
    shed = await call_asgi(
        app, "POST", CHAT, json_body=chat(800, 100), headers={"x-tenant-id": "a"}
    )
    assert shed.status == 429
    assert shed.headers["content-type"] == "application/json"
    assert shed.headers["x-admitq-shed-reason"] == "tenant_tokens"
    assert shed.headers["x-ratelimit-remaining-tokens"] == "0"
    error = shed.json()["error"]
    assert error["type"] == "rate_limit_error"
    assert error["code"] == "rate_limit_exceeded"
    assert error["reason"] == "tenant_tokens"
    assert "budget" in error["message"]
    assert int(shed.headers["retry-after"]) == pytest.approx(error["retry_after"], abs=1)
    assert backend.received == 1  # the shed request never reached the engine
    other = await call_asgi(
        app, "POST", CHAT, json_body=chat(800, 100), headers={"x-tenant-id": "b"}
    )
    assert other.status == 200


@virtual
async def test_retry_after_on_budget_shed_is_exact():
    backend = FakeModelBackend(report_usage=False)
    app = InferenceAdmission(backend, tenant_tpm=600, tenant_key=tenant_header)  # 10 tokens/s
    body = chat(96, 100)  # estimate: 96 + 4 + 100 = 200
    for _ in range(3):
        assert (await call_asgi(app, "POST", CHAT, json_body=body)).status == 200
    shed = await call_asgi(app, "POST", CHAT, json_body=body)
    assert shed.status == 429
    wait = shed.json()["error"]["retry_after"]
    await asyncio.sleep(wait - 0.01)
    assert (await call_asgi(app, "POST", CHAT, json_body=body)).status == 429
    await asyncio.sleep(0.5)
    assert (await call_asgi(app, "POST", CHAT, json_body=body)).status == 200


@virtual
async def test_overload_shed_is_a_503_and_queue_time_is_reported():
    backend = FakeModelBackend(stream_tps=50)
    app = InferenceAdmission(backend, max_inflight_requests=1, max_queue_s=1.0, early_shed=False)
    loop = asyncio.get_running_loop()
    first = asyncio.create_task(call_asgi(app, "POST", CHAT, json_body=chat(0, 25)))  # 0.5 s
    await settle()
    second = asyncio.create_task(call_asgi(app, "POST", CHAT, json_body=chat(0, 100)))  # 2 s
    third = asyncio.create_task(call_asgi(app, "POST", CHAT, json_body=chat(0, 10)))
    a, b, c = await asyncio.gather(first, second, third)
    assert (a.status, b.status, c.status) == (200, 200, 503)
    assert a.headers["x-admitq-queue-ms"] == "0"
    assert b.headers["x-admitq-queue-ms"] == "500"
    assert c.headers["x-admitq-shed-reason"] == "queue_timeout"
    assert c.duration == pytest.approx(1.0)
    assert c.json()["error"]["code"] == "server_overloaded"
    assert int(c.headers["retry-after"]) >= 1
    assert backend.received == 2
    assert loop.time() == pytest.approx(2.5)
    assert idle(app)


@virtual
async def test_client_timeout_header_bounds_the_queue_wait():
    backend = FakeModelBackend()
    app = InferenceAdmission(backend, max_inflight_requests=1, max_queue_s=30.0, early_shed=False)
    holder = asyncio.create_task(call_asgi(app, "POST", CHAT, json_body=chat(0, 500)))  # 10 s
    await settle()
    for header, value, expected in (
        ("x-request-timeout", "0.5", 0.5),
        ("x-stainless-read-timeout", "2", 2.0),
        ("x-envoy-expected-rq-timeout-ms", "250", 0.25),
        ("X-Request-Timeout", "garbage", 9.25),  # ignored: waits for the holder instead
    ):
        response = await call_asgi(app, "POST", CHAT, json_body=chat(0, 1), headers={header: value})
        if expected < 9:
            assert response.status == 503, header
            assert response.duration == pytest.approx(expected), header
        else:
            assert response.status == 200
    await holder


@virtual
async def test_response_headers_can_be_disabled():
    app = InferenceAdmission(
        FakeModelBackend(), tenant_tpm=100, response_headers=False, tenant_key=tenant_header
    )
    ok = await call_asgi(app, "POST", CHAT, json_body=chat(10, 10))
    assert not [h for h in ok.headers if h.startswith(("x-admitq", "x-ratelimit"))]
    shed = await call_asgi(app, "POST", CHAT, json_body=chat(100, 10))
    assert shed.status == 429
    assert "retry-after" in shed.headers  # that one is not optional
    assert "x-ratelimit-limit-tokens" not in shed.headers


# ------------------------------------------------------------------ usage correction


@virtual
async def test_non_streaming_usage_corrects_the_budget():
    backend = FakeModelBackend()
    app = InferenceAdmission(backend, tenant_tpm=10_000, tenant_key=tenant_header)
    body = chat(100, 4000, x_fake_output_tokens=10)  # asks for 4000, uses 10
    first = await call_asgi(app, "POST", CHAT, json_body=body)
    assert first.json()["usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110,
    }
    assert first.headers["x-ratelimit-remaining-tokens"] == str(10_000 - 4104)
    second = await call_asgi(app, "POST", CHAT, json_body=body)
    # 4104 were reserved for the first request but only 110 charged (ignore refill).
    remaining = int(second.headers["x-ratelimit-remaining-tokens"])
    assert remaining == pytest.approx(10_000 - 110 - 4104, abs=40)
    metrics = app.controller.metrics
    assert metrics.value("admitq_actual_tokens_total", tenant="anonymous", source="usage") == 220
    assert metrics.histogram("admitq_estimate_ratio", source="usage")[0] == 2


@virtual
async def test_streaming_usage_chunk_corrects_the_budget():
    backend = FakeModelBackend()
    app = InferenceAdmission(backend, tenant_tpm=10_000, tenant_key=tenant_header)
    body = chat(
        100, 2000, stream=True, stream_options={"include_usage": True}, x_fake_output_tokens=30
    )
    response = await call_asgi(app, "POST", CHAT, json_body=body)
    events = response.sse_events()
    assert len(events) == 31
    assert events[-1]["usage"]["completion_tokens"] == 30
    assert response.body.endswith(b"data: [DONE]\n\n")
    metrics = app.controller.metrics
    assert metrics.value("admitq_actual_tokens_total", tenant="anonymous", source="usage") == 130
    assert idle(app)


@virtual
async def test_streaming_without_usage_counts_deltas():
    backend = FakeModelBackend()
    app = InferenceAdmission(backend, tenant_tpm=10_000, tenant_key=tenant_header)
    body = chat(100, 2000, stream=True, x_fake_output_tokens=30)
    response = await call_asgi(app, "POST", CHAT, json_body=body)
    assert all("usage" not in event for event in response.sse_events())
    metrics = app.controller.metrics
    # Prompt: the estimate (100 + 4 overhead).  Output: 30 counted deltas.
    assert metrics.value("admitq_actual_tokens_total", tenant="anonymous", source="counted") == 134


@virtual
async def test_inject_stream_usage_rewrites_the_request():
    backend = FakeModelBackend()
    app = InferenceAdmission(backend, max_inflight_tokens=10**6, inject_stream_usage=True)
    body = chat(10, 50, stream=True, x_fake_output_tokens=5)
    response = await call_asgi(app, "POST", CHAT, json_body=body)
    assert backend.bodies[-1]["stream_options"] == {"include_usage": True}
    assert backend.bodies[-1]["messages"] == body["messages"]
    assert response.sse_events()[-1]["usage"]["completion_tokens"] == 5
    assert app.controller.metrics.total("admitq_actual_tokens_total", source="usage") == 15

    # Existing options are preserved; non-streaming requests are left alone.
    body = chat(10, 50, stream=True, stream_options={"continuous_usage_stats": True})
    await call_asgi(app, "POST", CHAT, json_body=body)
    assert backend.bodies[-1]["stream_options"] == {
        "continuous_usage_stats": True,
        "include_usage": True,
    }
    await call_asgi(app, "POST", CHAT, json_body=chat(10, 5))
    assert "stream_options" not in backend.bodies[-1]


@virtual
async def test_injected_body_gets_a_correct_content_length():
    downstream = Recorder()
    app = InferenceAdmission(downstream, max_inflight_tokens=10**6, inject_stream_usage=True)
    await call_asgi(app, "POST", CHAT, json_body=chat(10, 5, stream=True))
    headers = downstream.scopes[0]["headers"]
    lengths = [value for name, value in headers if name == b"content-length"]
    assert lengths == [str(len(downstream.body)).encode()]
    assert json.loads(downstream.body)["stream_options"] == {"include_usage": True}


@virtual
async def test_downstream_errors_are_not_charged():
    async def failing(scope, receive, send):
        await receive()
        body = b'{"error": {"message": "context length exceeded"}}'
        await send(
            {
                "type": "http.response.start",
                "status": 400,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})

    app = InferenceAdmission(failing, tenant_tpm=1000, tenant_key=tenant_header)
    for _ in range(5):  # would run out of budget after one if failures were charged
        response = await call_asgi(app, "POST", CHAT, json_body=chat(900, 50))
        assert response.status == 400
    metrics = app.controller.metrics
    assert metrics.value("admitq_completed_total", tenant="anonymous", outcome="client_error") == 5
    assert metrics.total("admitq_actual_tokens_total") == 0


@virtual
async def test_embeddings_reserve_no_output_tokens():
    backend = FakeModelBackend()
    app = InferenceAdmission(backend, max_inflight_tokens=10**6)
    response = await call_asgi(
        app, "POST", "/v1/embeddings", json_body={"input": "x" * 400, "max_tokens": 999}
    )
    assert response.status == 200
    assert response.headers["x-admitq-estimated-tokens"] == "100"


# ------------------------------------------------------------------ lifecycle


@virtual
async def test_capacity_is_held_for_the_whole_stream():
    backend = FakeModelBackend(stream_tps=10)
    app = InferenceAdmission(backend, max_inflight_tokens=1000)
    task = asyncio.create_task(
        call_asgi(app, "POST", CHAT, json_body=chat(96, 100, stream=True, x_fake_output_tokens=50))
    )
    for _ in range(4):
        await asyncio.sleep(1.0)  # mid-stream
        assert app.controller.snapshot().inflight_tokens == 200
        assert app.controller.snapshot().inflight_requests == 1
    response = await task
    assert response.completed
    assert response.chunks == 51
    assert idle(app)
    assert app.controller.metrics.histogram("admitq_time_to_first_byte_seconds")[0] == 1


@virtual
async def test_client_disconnect_mid_stream_releases_once_and_charges_partial_usage():
    backend = FakeModelBackend(stream_tps=10)
    app = InferenceAdmission(
        backend, max_inflight_tokens=1000, tenant_tpm=100_000, tenant_key=tenant_header
    )
    body = chat(96, 200, stream=True, x_fake_output_tokens=200)
    response = await call_asgi(app, "POST", CHAT, json_body=body, disconnect_after_chunks=7)
    assert response.disconnected
    assert not response.completed
    assert backend.aborted == 1  # the engine stopped generating
    assert backend.generated_tokens <= 9
    assert idle(app)
    metrics = app.controller.metrics
    assert (
        metrics.value("admitq_completed_total", tenant="anonymous", outcome="client_disconnect")
        == 1
    )
    assert metrics.total("admitq_completed_total") == 1  # released exactly once
    # Charged for the prompt and for what was streamed before the disconnect - not for
    # the 200 tokens that were reserved.
    charged = metrics.value("admitq_actual_tokens_total", tenant="anonymous", source="counted")
    assert 100 + 7 <= charged <= 100 + 9


@virtual
async def test_client_timeout_before_first_token_charges_nothing():
    backend = FakeModelBackend(prefill_tps=10)  # 96-token prompt takes 9.6 s to prefill
    app = InferenceAdmission(backend, max_inflight_tokens=1000)
    response = await call_asgi(app, "POST", CHAT, json_body=chat(96, 50), disconnect_after_s=2.0)
    assert response.status == 0
    assert response.disconnected
    assert backend.aborted == 1
    assert idle(app)
    assert app.controller.metrics.total("admitq_actual_tokens_total") == 0
    assert app.controller.metrics.total("admitq_completed_total", outcome="client_disconnect") == 1


@virtual
async def test_disconnect_while_queued_withdraws_the_request():
    backend = FakeModelBackend(stream_tps=10)
    app = InferenceAdmission(
        backend,
        max_inflight_requests=1,
        max_queue_s=60.0,
        tenant_tpm=1000,
        tenant_key=tenant_header,
    )
    holder = asyncio.create_task(
        call_asgi(app, "POST", CHAT, json_body=chat(0, 100), headers={"x-tenant-id": "h"})
    )
    await settle()
    queued = await call_asgi(app, "POST", CHAT, json_body=chat(500, 400), disconnect_after_s=3.0)
    assert queued.status == 0  # nothing was sent: there is nobody to send it to
    assert queued.disconnected
    assert queued.duration == pytest.approx(3.0)
    assert backend.received == 1  # never reached the engine
    assert app.controller.snapshot().queued_requests == 0
    await holder
    assert idle(app)
    metrics = app.controller.metrics
    assert metrics.total("admitq_completed_total", outcome="abandoned_in_queue") == 1
    # Its budget reservation came back: the same tenant can spend it again.
    again = await call_asgi(app, "POST", CHAT, json_body=chat(500, 400))
    assert again.status == 200


@virtual
async def test_app_exception_before_and_during_the_response_releases_capacity():
    async def explode_early(scope, receive, send):
        await receive()
        raise RuntimeError("early")

    async def explode_late(scope, receive, send):
        await receive()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        event = b'data: {"choices": [{"delta": {"content": "tok "}}]}\n\n'
        await send({"type": "http.response.body", "body": event, "more_body": True})
        raise RuntimeError("late")

    for downstream, charged in ((explode_early, 0), (explode_late, 105)):
        app = InferenceAdmission(downstream, max_inflight_tokens=1000)
        with pytest.raises(RuntimeError):
            await call_asgi(app, "POST", CHAT, json_body=chat(100, 50))
        assert idle(app)
        metrics = app.controller.metrics
        assert metrics.total("admitq_completed_total", outcome="error") == 1
        assert metrics.total("admitq_actual_tokens_total") == charged


@virtual
async def test_app_returning_without_a_response_releases_capacity():
    async def silent(scope, receive, send):
        await receive()

    app = InferenceAdmission(silent, max_inflight_tokens=1000)
    response = await call_asgi(app, "POST", CHAT, json_body=chat())
    assert response.status == 0
    assert idle(app)
    assert app.controller.metrics.total("admitq_completed_total", outcome="error") == 1


@virtual
async def test_cancelling_the_request_task_mid_stream_releases_capacity():
    backend = FakeModelBackend(stream_tps=10)
    app = InferenceAdmission(backend, max_inflight_tokens=1000)
    task = asyncio.create_task(
        call_asgi(app, "POST", CHAT, json_body=chat(96, 100, stream=True, x_fake_output_tokens=100))
    )
    await asyncio.sleep(2.0)
    assert app.controller.snapshot().inflight_requests == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert idle(app)
    assert backend.running == 0
    assert app.controller.metrics.total("admitq_completed_total", outcome="cancelled") == 1


@virtual
async def test_cancelling_the_request_task_while_queued_withdraws_it():
    backend = FakeModelBackend(stream_tps=10)
    app = InferenceAdmission(backend, max_inflight_requests=1, max_queue_s=60.0)
    holder = asyncio.create_task(call_asgi(app, "POST", CHAT, json_body=chat(0, 50)))
    await settle()
    queued = asyncio.create_task(call_asgi(app, "POST", CHAT, json_body=chat(0, 50)))
    await settle()
    assert app.controller.snapshot().queued_requests == 1
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    assert app.controller.snapshot().queued_requests == 0
    await holder
    assert idle(app)
    assert backend.received == 1


@virtual
async def test_cancel_on_disconnect_stops_a_downstream_that_does_not_listen():
    # A backend that keeps generating after the client has gone (think: a proxy that does
    # not forward disconnects).  Without propagation the capacity stays held until the
    # whole completion has been generated; with it, the app is cancelled at once.
    results = {}
    for propagate in (False, True):
        backend = FakeModelBackend(stream_tps=10, abort_on_disconnect=False)
        app = InferenceAdmission(backend, max_inflight_tokens=1000, cancel_on_disconnect=propagate)
        body = chat(96, 100, stream=True, x_fake_output_tokens=100)  # 10 s of output
        response = await call_asgi(app, "POST", CHAT, json_body=body, disconnect_after_s=2.0)
        assert response.disconnected
        assert idle(app)
        assert backend.running == 0
        results[propagate] = (response.duration, backend.generated_tokens)
        outcome = app.controller.metrics.total(
            "admitq_completed_total", outcome="client_disconnect"
        )
        assert outcome == (1 if propagate else 0)
    assert results[False][0] == pytest.approx(10.0, abs=0.2)
    assert results[False][1] == 100
    assert results[True][0] == pytest.approx(2.0, abs=0.2)
    assert results[True][1] <= 21


@virtual
async def test_cancel_on_disconnect_grace_period_lets_the_app_finish():
    backend = FakeModelBackend(stream_tps=10, abort_on_disconnect=False)
    app = InferenceAdmission(
        backend, max_inflight_tokens=1000, cancel_on_disconnect=True, disconnect_grace_s=60.0
    )
    body = chat(96, 100, stream=True, x_fake_output_tokens=30)  # 3 s of output
    response = await call_asgi(app, "POST", CHAT, json_body=body, disconnect_after_s=1.0)
    assert response.duration == pytest.approx(3.0, abs=0.2)
    assert backend.completed == 1
    assert idle(app)


@virtual
async def test_cancel_on_disconnect_does_not_disturb_normal_requests():
    backend = FakeModelBackend()
    app = InferenceAdmission(backend, max_inflight_tokens=1000, cancel_on_disconnect=True)
    for stream in (False, True):
        response = await call_asgi(app, "POST", CHAT, json_body=chat(96, 20, stream=stream))
        assert response.status == 200
        assert response.completed
    assert idle(app)
    assert app.controller.metrics.total("admitq_completed_total", outcome="ok") == 2

    async def explode(scope, receive, send):
        raise RuntimeError("boom")

    app = InferenceAdmission(explode, max_inflight_tokens=1000, cancel_on_disconnect=True)
    with pytest.raises(RuntimeError, match="boom"):
        await call_asgi(app, "POST", CHAT, json_body=chat())
    assert idle(app)


@virtual
async def test_no_response_is_ever_shed_after_it_has_started():
    # Saturate the controller while a stream is in flight; the stream must run to
    # completion untouched even though later arrivals are being shed all around it.
    backend = FakeModelBackend(stream_tps=20)
    app = InferenceAdmission(backend, max_inflight_requests=1, max_queue_s=0.2)
    stream = asyncio.create_task(
        call_asgi(app, "POST", CHAT, json_body=chat(0, 100, stream=True, x_fake_output_tokens=100))
    )
    await settle()
    statuses = []
    for _ in range(20):  # 20 x (0.2 s queue + 0.01 s) = 4.2 s, inside the 5 s stream
        statuses.append((await call_asgi(app, "POST", CHAT, json_body=chat(0, 1))).status)
        await asyncio.sleep(0.01)
    response = await stream
    assert set(statuses) == {503}
    assert response.status == 200
    assert response.completed
    assert response.chunks == 101
    assert response.body.count(b"data: ") == 101


# ------------------------------------------------------------------ configuration


@virtual
async def test_tenant_and_priority_keys():
    backend = FakeModelBackend()
    app = InferenceAdmission(
        backend,
        max_inflight_tokens=10**6,
        tenant_key=header_tenant("X-Org"),
        priority_key=lambda info: (info.body or {}).get("service_tier"),
        priorities={"interactive": 0, "batch": 1},
    )
    await call_asgi(app, "POST", CHAT, json_body=chat(), headers={"x-org": "acme"})
    await call_asgi(app, "POST", CHAT, json_body={**chat(), "service_tier": "batch"})
    metrics = app.controller.metrics
    assert metrics.value("admitq_admitted_total", tenant="acme", priority="interactive") == 1
    assert metrics.value("admitq_admitted_total", tenant="anonymous", priority="batch") == 1


@virtual
async def test_default_tenant_is_a_hash_of_the_api_key():
    app = InferenceAdmission(FakeModelBackend(), tenant_tpm=300)
    headers = {"authorization": "Bearer sk-secret-1"}
    assert (
        await call_asgi(app, "POST", CHAT, json_body=chat(196, 100), headers=headers)
    ).status == 200
    assert (
        await call_asgi(app, "POST", CHAT, json_body=chat(196, 100), headers=headers)
    ).status == 429
    other = {"authorization": "Bearer sk-secret-2"}
    assert (
        await call_asgi(app, "POST", CHAT, json_body=chat(196, 100), headers=other)
    ).status == 200
    rendered = app.controller.metrics.render()
    assert "sk-secret" not in rendered
    assert 'tenant="key-' in rendered


@virtual
async def test_custom_routes_and_mount_prefixes():
    backend = Recorder()
    app = InferenceAdmission(
        backend,
        max_inflight_tokens=10**6,
        routes={
            "/generate": "other",
            r"^/models/[^/]+/infer": "other",
            "/v1/embeddings": "embeddings",
        },
    )
    for path, admitted in (
        ("/generate", True),
        ("/api/generate/", True),
        ("/models/llama/infer", True),
        ("/models/llama/other", False),
        ("/v1/chat/completions", False),  # replaced by the custom table
    ):
        response = await call_asgi(app, "POST", path, json_body={"prompt": "hello", "n_predict": 8})
        assert ("x-admitq-estimated-tokens" in response.headers) is admitted, path
    # Default table matches under any mount prefix.
    app = InferenceAdmission(Recorder(), max_inflight_tokens=10**6)
    response = await call_asgi(app, "POST", "/openai/v1/chat/completions", json_body=chat())
    assert "x-admitq-estimated-tokens" in response.headers


@virtual
async def test_metrics_path_and_shared_controller():
    controller = AdmissionController(max_inflight_tokens=10**6)
    one = InferenceAdmission(
        FakeModelBackend(), controller=controller, metrics_path="/admitq/metrics"
    )
    two = InferenceAdmission(FakeModelBackend(), controller=controller)
    await call_asgi(one, "POST", CHAT, json_body=chat())
    await call_asgi(two, "POST", CHAT, json_body=chat())
    scrape = await call_asgi(one, "GET", "/admitq/metrics")
    assert scrape.status == 200
    assert scrape.headers["content-type"].startswith("text/plain; version=0.0.4")
    assert b'admitq_admitted_total{tenant="anonymous",priority="interactive"} 2' in scrape.body
    # Without metrics_path the request falls through to the app (whose own /metrics this is).
    passed_on = await call_asgi(two, "GET", "/admitq/metrics")
    assert b"vllm:num_requests_waiting" in passed_on.body


def test_constructor_validation():
    async def downstream(scope, receive, send):
        pass

    with pytest.raises(TypeError, match="not both"):
        InferenceAdmission(downstream, controller=AdmissionController(), max_queue_s=1.0)
    with pytest.raises(ValueError, match="max_body_bytes"):
        InferenceAdmission(downstream, max_body_bytes=-1)
    with pytest.raises(ValueError, match="endpoint kind"):
        InferenceAdmission(downstream, routes={"/x": "bogus"})
    with pytest.raises(ValueError, match="must start with"):
        RouteMatcher({"x": "chat"})
    matcher = RouteMatcher(["/v1/chat/completions", "/custom"], methods=("POST", "put"))
    assert matcher.match("PUT", "/custom") == "other"
    assert matcher.match("post", "/v1/chat/completions") == "chat"
    assert matcher.match("GET", "/custom") is None


def test_middleware_accepts_every_controller_option():
    """The middleware spells out AdmissionConfig's fields; keep the two in step."""
    import dataclasses
    import inspect

    from admitq import AdmissionConfig

    parameters = inspect.signature(InferenceAdmission.__init__).parameters
    for field in dataclasses.fields(AdmissionConfig):
        assert field.name in parameters, field.name
        parameter = parameters[field.name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        if field.default is not dataclasses.MISSING and field.name != "priorities":
            assert parameter.default == field.default, field.name
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


def test_controller_and_options_conflict_names_the_options():
    async def downstream(scope, receive, send):
        pass

    with pytest.raises(TypeError, match="max_queue_s, tenant_tpm"):
        InferenceAdmission(
            downstream, controller=AdmissionController(), max_queue_s=1.0, tenant_tpm=5
        )
    # Middleware-only options are fine next to a controller.
    InferenceAdmission(downstream, controller=AdmissionController(), metrics_path="/m")

"""End-to-end over real sockets: a uvicorn server, httpx and raw TCP clients.

These are the only tests that run in real time, so they are kept short.  They exist for
the one thing in-process tests cannot show: what happens when a real client really goes
away mid-stream.
"""

import asyncio
import contextlib
import json
import random
import socket

import pytest

httpx = pytest.importorskip("httpx")
uvicorn = pytest.importorskip("uvicorn")

from admitq import InferenceAdmission  # noqa: E402
from admitq.testing import FakeModelBackend  # noqa: E402

CHAT = "/v1/chat/completions"


def chat(prompt_tokens=50, max_tokens=20, **extra):
    return {
        "model": "m",
        "messages": [{"role": "user", "content": "x" * (4 * prompt_tokens)}],
        "max_tokens": max_tokens,
        **extra,
    }


@contextlib.asynccontextmanager
async def serve(app):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    config = uvicorn.Config(app, log_level="warning", lifespan="off", access_log=False)
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        for _ in range(500):
            if server.started:
                break
            await asyncio.sleep(0.01)
        assert server.started
        yield f"http://127.0.0.1:{port}", port
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=10)


async def wait_idle(app: InferenceAdmission, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        snap = app.controller.snapshot()
        if (snap.inflight_requests, snap.inflight_tokens, snap.queued_requests) == (0, 0, 0):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"capacity leaked: {app.controller.snapshot()}")


async def test_stream_end_to_end_over_http():
    backend = FakeModelBackend(stream_tps=500, total_tps=5000)
    app = InferenceAdmission(backend, max_inflight_tokens=10_000, tenant_tpm=100_000)
    async with serve(app) as (base, _), httpx.AsyncClient(base_url=base) as client:
        body = chat(50, 40, stream=True, stream_options={"include_usage": True})
        async with client.stream("POST", CHAT, json=body) as response:
            assert response.status_code == 200
            assert response.headers["x-admitq-estimated-tokens"] == "94"
            assert "x-ratelimit-remaining-tokens" in response.headers
            lines = [line async for line in response.aiter_lines() if line.startswith("data: ")]
        assert lines[-1] == "data: [DONE]"
        events = [json.loads(line[6:]) for line in lines[:-1]]
        assert events[-1]["usage"] == {
            "prompt_tokens": 50,
            "completion_tokens": 40,
            "total_tokens": 90,
        }
        plain = await client.post(CHAT, json=chat(50, 10))
        assert plain.json()["usage"]["completion_tokens"] == 10
        await wait_idle(app)
        metrics = app.controller.metrics
        assert metrics.total("admitq_completed_total", outcome="ok") == 2
        assert metrics.total("admitq_actual_tokens_total", source="usage") == 90 + 60


async def test_client_disconnect_mid_stream_over_http_releases_capacity():
    backend = FakeModelBackend(stream_tps=50)
    app = InferenceAdmission(backend, max_inflight_tokens=10_000)
    async with serve(app) as (base, _), httpx.AsyncClient(base_url=base) as client:
        body = chat(50, 2000, stream=True, x_fake_output_tokens=2000)  # would take 40 s
        async with client.stream("POST", CHAT, json=body) as response:
            seen = 0
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    seen += 1
                if seen == 5:
                    break  # leaving the block closes the connection mid-stream
        assert app.controller.snapshot().inflight_requests in (0, 1)
        await wait_idle(app)
        assert backend.aborted == 1
        assert backend.running == 0
        assert backend.generated_tokens < 200
        metrics = app.controller.metrics
        assert metrics.total("admitq_completed_total") == 1  # released exactly once
        assert metrics.total("admitq_completed_total", outcome="client_disconnect") == 1
        charged = metrics.total("admitq_actual_tokens_total", source="counted")
        assert 54 + 5 <= charged < 54 + 200  # prompt estimate + tokens streamed so far


async def test_abrupt_tcp_close_releases_capacity():
    backend = FakeModelBackend(stream_tps=50)
    app = InferenceAdmission(backend, max_inflight_tokens=10_000)
    async with serve(app) as (_, port):
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        payload = json.dumps(chat(50, 2000, stream=True)).encode()
        writer.write(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(payload)}\r\n\r\n".encode()
            + payload
        )
        await writer.drain()
        data = b""
        while data.count(b"data: ") < 3:
            data += await reader.read(4096)
        assert app.controller.snapshot().inflight_requests == 1
        # Reset the connection instead of closing it politely.
        writer.get_extra_info("socket").setsockopt(
            socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00"
        )
        writer.close()
        await wait_idle(app)
        assert backend.aborted == 1
        assert app.controller.metrics.total("admitq_completed_total") == 1


async def test_disconnect_while_queued_over_http_never_reaches_the_engine():
    backend = FakeModelBackend(stream_tps=50)
    app = InferenceAdmission(backend, max_inflight_requests=1, max_queue_s=30.0, early_shed=False)
    async with serve(app) as (base, _), httpx.AsyncClient(base_url=base) as client:
        holder = asyncio.create_task(client.post(CHAT, json=chat(10, 60)))  # ~1.2 s
        while app.controller.snapshot().inflight_requests == 0:
            await asyncio.sleep(0.01)
        with pytest.raises(httpx.ReadTimeout):
            await client.post(CHAT, json=chat(10, 5), timeout=0.2)
        for _ in range(300):
            if app.controller.snapshot().queued_requests == 0:
                break
            await asyncio.sleep(0.01)
        assert app.controller.snapshot().queued_requests == 0
        assert (await holder).status_code == 200
        await wait_idle(app)
        assert backend.received == 1
        assert (
            app.controller.metrics.total("admitq_completed_total", outcome="abandoned_in_queue")
            == 1
        )


async def test_starlette_streaming_response_disconnect_runs_generator_cleanup():
    pytest.importorskip("starlette")
    from starlette.applications import Starlette
    from starlette.responses import StreamingResponse
    from starlette.routing import Route

    state = {"started": 0, "cleaned": 0, "yielded": 0}

    async def completions(request):
        await request.json()

        async def events():
            state["started"] += 1
            try:
                while True:
                    state["yielded"] += 1
                    yield b'data: {"choices": [{"delta": {"content": "tok "}}]}\n\n'
                    await asyncio.sleep(0.02)
            finally:
                state["cleaned"] += 1

        return StreamingResponse(events(), media_type="text/event-stream")

    inner = Starlette(routes=[Route(CHAT, completions, methods=["POST"])])
    for propagate in (False, True):
        app = InferenceAdmission(inner, max_inflight_requests=4, cancel_on_disconnect=propagate)
        async with serve(app) as (base, _), httpx.AsyncClient(base_url=base) as client:
            async with client.stream("POST", CHAT, json=chat(stream=True)) as response:
                async for _ in response.aiter_lines():
                    if state["yielded"] >= 5:
                        break
            await wait_idle(app)
        assert (
            app.controller.metrics.total("admitq_completed_total", outcome="client_disconnect") == 1
        )
    assert state["started"] == 2
    assert state["cleaned"] == 2  # the endless generators were closed, not abandoned


@pytest.mark.parametrize("seed", [1, 2, 3])
async def test_real_concurrency_storm_does_not_leak(seed):
    rng = random.Random(seed)
    backend = FakeModelBackend(max_num_seqs=4, total_tps=4000, stream_tps=400)
    limit = 6
    peak = 0

    async def probe(scope, receive, send):
        nonlocal peak
        if scope["type"] == "http":
            peak = max(peak, app.controller.snapshot().inflight_requests)
            assert app.controller.snapshot().inflight_requests <= limit
        await backend(scope, receive, send)

    app = InferenceAdmission(
        probe,
        max_inflight_requests=limit,
        max_inflight_tokens=3000,
        max_queue_s=0.3,
        tenant_rpm=6000,
    )
    statuses = []

    async def one(client):
        await asyncio.sleep(rng.uniform(0, 0.5))
        body = chat(rng.randint(1, 200), rng.randint(1, 120), stream=rng.random() < 0.7)
        timeout = rng.choice([None, None, 0.05, 0.15])
        try:
            if body["stream"]:
                async with client.stream("POST", CHAT, json=body, timeout=timeout) as response:
                    cut = rng.choice([None, 3, 10])
                    count = 0
                    async for _ in response.aiter_lines():
                        count += 1
                        if cut is not None and count >= cut:
                            break
                    statuses.append(response.status_code)
            else:
                statuses.append((await client.post(CHAT, json=body, timeout=timeout)).status_code)
        except httpx.TimeoutException:
            statuses.append(-1)

    limits = httpx.Limits(max_connections=200)
    async with serve(app) as (base, _), httpx.AsyncClient(base_url=base, limits=limits) as client:
        await asyncio.gather(*(one(client) for _ in range(150)))
        await wait_idle(app)
    assert backend.running == 0
    assert backend.waiting == 0
    assert peak <= limit
    metrics = app.controller.metrics
    assert metrics.total("admitq_admitted_total") == metrics.total("admitq_completed_total") - (
        metrics.total("admitq_completed_total", outcome="abandoned_in_queue")
    )
    assert 200 in statuses
    assert 503 in statuses

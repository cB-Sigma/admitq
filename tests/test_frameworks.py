"""The middleware inside real frameworks, driven through httpx's ASGI transport."""

import asyncio
import json

import pytest

httpx = pytest.importorskip("httpx")

from admitq import AdmissionController, InferenceAdmission, header_tenant, metrics_app  # noqa: E402

CHAT = "/v1/chat/completions"


def client_for(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def chat(prompt_tokens: int = 100, max_tokens: int = 20, **extra):
    return {
        "model": "m",
        "messages": [{"role": "user", "content": "x" * (4 * prompt_tokens)}],
        "max_tokens": max_tokens,
        **extra,
    }


def sse(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


# ------------------------------------------------------------------ FastAPI


async def _tag(request, call_next):
    response = await call_next(request)
    response.headers["x-tagged"] = "1"
    return response


def build_fastapi(*, base_http_middleware: str | None = None, **options):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    from starlette.middleware.base import BaseHTTPMiddleware

    class Message(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str
        messages: list[Message]
        max_tokens: int = 16
        stream: bool = False

    app = fastapi.FastAPI()
    # Starlette wraps in reverse order of registration: the last one added is outermost.
    if base_http_middleware == "inside":
        app.add_middleware(BaseHTTPMiddleware, dispatch=_tag)
    # The documented one-liner.
    app.add_middleware(InferenceAdmission, **options)
    if base_http_middleware == "outside":
        app.add_middleware(BaseHTTPMiddleware, dispatch=_tag)
    seen: list[ChatRequest] = []

    @app.post(CHAT)
    async def chat_completions(request: ChatRequest):
        seen.append(request)
        prompt = sum(len(m.content) for m in request.messages) // 4
        if request.stream:

            async def events():
                for _ in range(request.max_tokens):
                    yield sse({"choices": [{"index": 0, "delta": {"content": "tok "}}]})
                    await asyncio.sleep(0)
                usage = {"prompt_tokens": prompt, "completion_tokens": request.max_tokens}
                yield sse({"choices": [], "usage": usage})
                yield b"data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")
        return {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": 2, "total_tokens": prompt + 2},
        }

    @app.get("/health")
    async def health():
        return {"ok": True}

    return app, seen


async def test_fastapi_one_liner_body_replay_and_headers():
    app, seen = build_fastapi(
        max_queue_s=2.0, max_inflight_tokens=64_000, tenant_tpm={"default": 200_000}
    )
    async with client_for(app) as client:
        response = await client.post(
            CHAT, json=chat(100, 20), headers={"authorization": "Bearer k1"}
        )
        assert response.status_code == 200
        assert response.json()["usage"]["prompt_tokens"] == 100
        # FastAPI parsed and validated the replayed body.
        assert seen[0].messages[0].content == "x" * 400
        assert seen[0].max_tokens == 20
        assert response.headers["x-admitq-queue-ms"] == "0"
        assert response.headers["x-admitq-estimated-tokens"] == "124"
        assert response.headers["x-ratelimit-limit-tokens"] == "200000"
        assert response.headers["x-ratelimit-remaining-tokens"] == str(200_000 - 124)
        # Validation errors still come from FastAPI, untouched.
        invalid = await client.post(CHAT, json={"model": "m"})
        assert invalid.status_code == 422
        not_json = await client.post(
            CHAT, content=b"not json", headers={"content-type": "text/plain"}
        )
        assert not_json.status_code == 422
        # Other routes do not go through admission at all.
        health = await client.get("/health")
        assert health.json() == {"ok": True}
        assert "x-admitq-queue-ms" not in health.headers


async def test_fastapi_streaming_response_and_budget_shed():
    app, _ = build_fastapi(tenant_tpm=1000, tenant_key=header_tenant("x-tenant"))
    async with client_for(app) as client:
        body = chat(100, 50, stream=True)
        response = await client.post(CHAT, json=body, headers={"x-tenant": "a"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.text.count("data: ") == 52
        assert response.text.endswith("data: [DONE]\n\n")
        # 154 estimated, 150 actually used; a 900-token request no longer fits.
        shed = await client.post(CHAT, json=chat(796, 100), headers={"x-tenant": "a"})
        assert shed.status_code == 429
        assert shed.json()["error"]["code"] == "rate_limit_exceeded"
        assert int(shed.headers["retry-after"]) >= 1
        ok = await client.post(CHAT, json=chat(796, 100), headers={"x-tenant": "b"})
        assert ok.status_code == 200


async def test_fastapi_large_body_beyond_the_buffer_cap():
    app, seen = build_fastapi(max_inflight_tokens=10**9, max_body_bytes=64 * 1024)
    async with client_for(app) as client:
        body = chat(prompt_tokens=300_000, max_tokens=5)  # a 1.2 MB JSON body
        response = await client.post(CHAT, json=body)
        assert response.status_code == 200
        assert len(seen[0].messages[0].content) == 1_200_000
        # Costed from the body size, since it was never parsed by admitq.
        assert int(response.headers["x-admitq-estimated-tokens"]) > 300_000


async def test_fastapi_queueing_under_concurrency():
    app, _ = build_fastapi(max_inflight_requests=2, max_queue_s=5.0)
    async with client_for(app) as client:
        responses = await asyncio.gather(
            *(client.post(CHAT, json=chat(10, 30, stream=True)) for _ in range(20))
        )
    assert [r.status_code for r in responses] == [200] * 20
    assert all(r.text.count("data: ") == 32 for r in responses)


@pytest.mark.parametrize("position", ["inside", "outside"])
async def test_works_inside_and_outside_base_http_middleware(position):
    app, seen = build_fastapi(base_http_middleware=position, max_inflight_tokens=10**6)
    async with client_for(app) as client:
        response = await client.post(CHAT, json=chat(100, 10, stream=True))
        assert response.status_code == 200
        assert response.headers["x-tagged"] == "1"
        assert response.headers["x-admitq-estimated-tokens"] == "114"
        assert response.text.count("data: ") == 12
        plain = await client.post(CHAT, json=chat(100, 10))
        assert plain.json()["usage"]["completion_tokens"] == 2
    assert len(seen) == 2


# ------------------------------------------------------------------ Starlette


async def test_starlette_with_shared_controller_and_mounted_metrics():
    pytest.importorskip("starlette")
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route

    controller = AdmissionController(max_inflight_tokens=10_000, tenant_rpm=2, burst_s=60)

    async def completions(request):
        payload = await request.json()
        return JSONResponse(
            {"echo": payload["prompt"], "usage": {"prompt_tokens": 3, "completion_tokens": 4}}
        )

    app = Starlette(
        routes=[
            Route("/v1/completions", completions, methods=["POST"]),
            Mount("/metrics", app=metrics_app(controller)),
        ],
        middleware=[Middleware(InferenceAdmission, controller=controller)],
    )
    async with client_for(app) as client:
        first = await client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 8})
        assert first.json()["echo"] == "hi"
        assert first.headers["x-ratelimit-limit-requests"] == "2"
        assert first.headers["x-ratelimit-remaining-requests"] == "1"
        assert (await client.post("/v1/completions", json={"prompt": "hi"})).status_code == 200
        third = await client.post("/v1/completions", json={"prompt": "hi"})
        assert third.status_code == 429
        assert third.headers["x-admitq-shed-reason"] == "tenant_requests"
        assert third.headers["x-ratelimit-remaining-requests"] == "0"
        scrape = await client.get("/metrics/")
        assert scrape.status_code == 200
        labels = 'tenant="anonymous",priority="interactive",reason="tenant_requests"'
        assert f"admitq_shed_total{{{labels}}} 1" in scrape.text
        assert 'admitq_actual_tokens_total{tenant="anonymous",source="usage"} 14' in scrape.text


# ------------------------------------------------------------------ Litestar


async def test_litestar():
    litestar = pytest.importorskip("litestar")
    from litestar.middleware import DefineMiddleware

    @litestar.post(CHAT, status_code=200)
    async def chat_completions(request: litestar.Request) -> dict:
        data = await request.json()
        return {"n": len(data["messages"]), "usage": {"prompt_tokens": 7, "completion_tokens": 1}}

    @litestar.get("/health")
    async def health() -> dict:
        return {"ok": True}

    app = litestar.Litestar(
        route_handlers=[chat_completions, health],
        middleware=[
            DefineMiddleware(
                InferenceAdmission, max_inflight_tokens=10_000, tenant_tpm=300, max_queue_s=1.0
            )
        ],
    )
    async with client_for(app) as client:
        ok = await client.post(CHAT, json=chat(100, 20))
        assert ok.status_code == 200
        assert ok.json()["n"] == 1
        assert ok.headers["x-admitq-estimated-tokens"] == "124"
        shed = await client.post(CHAT, json=chat(300, 20))
        assert shed.status_code == 429
        assert shed.json()["error"]["type"] == "rate_limit_error"
        health_response = await client.get("/health")
        assert health_response.status_code == 200
        assert "x-admitq-queue-ms" not in health_response.headers

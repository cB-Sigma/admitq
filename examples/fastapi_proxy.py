"""A FastAPI proxy in front of an OpenAI-compatible model server, protected by admitq.

Works with anything that speaks the OpenAI HTTP API: vLLM (``vllm serve``), llama.cpp
(``llama-server``), llama-cpp-python (``python -m llama_cpp.server``), SGLang, TGI's
OpenAI routes, Ollama (``/v1``).

    pip install "admitq @ git+https://github.com/cb-Sigma/admitq" fastapi uvicorn httpx
    UPSTREAM=http://127.0.0.1:8000 uvicorn examples.fastapi_proxy:app --port 9000

    curl -N http://127.0.0.1:9000/v1/chat/completions \
      -H 'authorization: Bearer my-key' -H 'content-type: application/json' \
      -d '{"model": "m", "stream": true, "messages": [{"role": "user", "content": "hi"}]}'

    curl http://127.0.0.1:9000/admitq/metrics

The proxy itself is deliberately small.  The interesting part is ``create_app``'s
``admission`` argument: everything admitq does is configured there.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from admitq import InferenceAdmission, PriorityClass

# Headers that describe one hop and must not be copied to the next.
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "transfer-encoding",
    "content-length",
    "content-encoding",
    "te",
    "upgrade",
}

DEFAULT_ADMISSION: dict[str, Any] = {
    # How much estimated work may be in flight.  Start from what the engine can hold
    # (vLLM: --max-num-seqs; llama.cpp: --parallel) and tune with the metrics.
    "max_inflight_requests": 32,
    "max_inflight_tokens": 64_000,
    # Nobody waits longer than this for capacity; beyond it the answer is a fast 503.
    "max_queue_s": 2.0,
    # Per-tenant budgets.  "default" applies to every tenant not listed.
    "tenant_tpm": {"default": 200_000},
    "tenant_rpm": {"default": 600},
    # No tenant may hold more than half of the in-flight tokens.
    "max_tenant_share": 0.5,
    # x-priority: batch waits up to 30 s and yields to interactive traffic.
    "priorities": [PriorityClass("interactive", 0), PriorityClass("batch", 1, max_queue_s=30.0)],
    # If the client hangs up while we wait for the engine, stop waiting.
    "cancel_on_disconnect": True,
    # Ask the engine for exact usage on streams (adds one final chunk for the client).
    "inject_stream_usage": False,
    "metrics_path": "/admitq/metrics",
}


def create_app(upstream: str, *, admission: dict[str, Any] | None = DEFAULT_ADMISSION) -> FastAPI:
    """Build the proxy.  Pass ``admission=None`` for a bare proxy without admitq."""

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        timeout = httpx.Timeout(connect=5.0, read=None, write=30.0, pool=None)
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=64)
        async with httpx.AsyncClient(base_url=upstream, timeout=timeout, limits=limits) as client:
            app.state.client = client
            yield

    app = FastAPI(title="admitq example proxy", lifespan=lifespan)
    if admission is not None:
        app.add_middleware(InferenceAdmission, **admission)

    async def forward(request: Request) -> Response:
        client: httpx.AsyncClient = request.app.state.client
        headers = {
            name: value
            for name, value in request.headers.items()
            if name in ("content-type", "authorization", "accept")
        }
        upstream_request = client.build_request(
            "POST", request.url.path, content=await request.body(), headers=headers
        )
        # Waiting for the engine's response headers can take as long as the engine's own
        # queue.  Do not keep an upstream request alive for a client that has gone.
        pending = asyncio.ensure_future(client.send(upstream_request, stream=True))
        try:
            while not pending.done():
                await asyncio.wait({pending}, timeout=0.25)
                if not pending.done() and await request.is_disconnected():
                    pending.cancel()
                    return Response(status_code=499)
            upstream_response = pending.result()
        except asyncio.CancelledError:
            pending.cancel()
            raise
        except httpx.HTTPError as exc:
            return Response(f"upstream error: {exc!r}", status_code=502)

        async def body() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream_response.aiter_raw():
                    yield chunk
            finally:
                # Runs on completion, on error and when the client disconnects: closing
                # the upstream response is what tells the engine to stop generating.
                await upstream_response.aclose()

        response_headers = {
            name: value
            for name, value in upstream_response.headers.items()
            if name.lower() not in HOP_BY_HOP
        }
        return StreamingResponse(
            body(), status_code=upstream_response.status_code, headers=response_headers
        )

    for path in ("/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/responses"):
        app.add_api_route(path, forward, methods=["POST"])

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        state: dict[str, Any] = {"status": "ok"}
        if admission is not None:
            # The middleware instance is created lazily by Starlette; report what we can.
            state["admission"] = sorted(admission)
        return state

    return app


app = create_app(os.environ.get("UPSTREAM", "http://127.0.0.1:8000"))

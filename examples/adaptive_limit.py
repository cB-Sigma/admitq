"""An in-flight limit that tunes itself from the engine's queue depth.

    pip install "admitq @ git+https://github.com/cb-Sigma/admitq" fastapi uvicorn httpx
    UPSTREAM=http://127.0.0.1:8000 uvicorn examples.adaptive_limit:app --port 9000

Two ways to drive :class:`admitq.AdaptiveLimit`:

* ``signal="ttfb"`` (default): admitq measures time to first byte itself.  Nothing else to
  set up.
* ``signal="reported"``: you feed it a number from the engine.  Below, vLLM's
  ``vllm:num_requests_waiting`` gauge is polled once a second: "keep fewer than two
  requests waiting inside the engine".

Swap ``vllm_requests_waiting`` for ``llamacpp_requests_deferred`` (llama-server started
with ``--metrics``), ``sglang_queue_requests`` or ``tgi_queue_size`` for other engines.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from admitq import AdaptiveLimit, AdmissionController, InferenceAdmission
from admitq.signals import poll_signal, vllm_requests_waiting

UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:8000")

controller = AdmissionController(
    max_inflight_requests=16,  # the starting point
    max_queue_s=2.0,
    adaptive=AdaptiveLimit(
        signal="reported",
        target=2.0,  # at most two requests waiting inside the engine
        min_limit=2,
        max_limit=256,
        dimension="requests",
        interval_s=10.0,
    ),
)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with httpx.AsyncClient(base_url=UPSTREAM, timeout=None) as client:
        app.state.client = client

        async def fetch() -> str:
            return (await client.get("/metrics", timeout=2.0)).text

        poller = asyncio.create_task(
            poll_signal(controller, fetch, parse=vllm_requests_waiting, interval_s=1.0)
        )
        try:
            yield
        finally:
            poller.cancel()
            await controller.aclose()


app = FastAPI(lifespan=lifespan)
app.add_middleware(InferenceAdmission, controller=controller, metrics_path="/admitq/metrics")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> StreamingResponse:
    client: httpx.AsyncClient = request.app.state.client
    upstream = await client.send(
        client.build_request(
            "POST",
            "/v1/chat/completions",
            content=await request.body(),
            headers={"content-type": "application/json"},
        ),
        stream=True,
    )

    async def body() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        body(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
    )

"""admitq in a Litestar application.

    pip install "admitq @ git+https://github.com/cb-Sigma/admitq" litestar uvicorn
    uvicorn examples.litestar_app:app --port 9000

The middleware is pure ASGI, so Litestar's ``DefineMiddleware`` is all it takes.  The
handler below fakes a model so the example runs on its own.
"""

from __future__ import annotations

import asyncio
from typing import Any

from litestar import Litestar, Request, get, post
from litestar.middleware import DefineMiddleware

from admitq import AdmissionController, InferenceAdmission, header_tenant

# Build the controller yourself when you want a handle on it (for /stats, tests, ...).
controller = AdmissionController(
    max_inflight_requests=4,
    max_inflight_tokens=8_000,
    max_queue_s=1.5,
    tenant_tpm={"default": 30_000, "internal": 300_000},
    tenant_weights={"internal": 4},
)


@post("/v1/chat/completions", status_code=200)
async def chat_completions(request: Request[Any, Any, Any]) -> dict[str, Any]:
    body = await request.json()
    max_tokens = int(body.get("max_tokens", 32))
    await asyncio.sleep(0.02 * max_tokens)  # pretend to generate
    return {
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "tok " * max_tokens}}],
        # admitq reads this to correct its estimate.
        "usage": {"prompt_tokens": 12, "completion_tokens": max_tokens},
    }


@get("/stats")
async def stats() -> dict[str, Any]:
    snapshot = controller.snapshot()
    return {
        "inflight_requests": snapshot.inflight_requests,
        "inflight_tokens": snapshot.inflight_tokens,
        "queued_requests": snapshot.queued_requests,
        "mean_hold_s": snapshot.mean_hold_s,
    }


app = Litestar(
    route_handlers=[chat_completions, stats],
    middleware=[
        DefineMiddleware(
            InferenceAdmission,
            controller=controller,
            tenant_key=header_tenant("x-tenant-id"),
            metrics_path="/admitq/metrics",
        )
    ],
)

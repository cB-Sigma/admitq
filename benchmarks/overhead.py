"""Per-request overhead of the middleware itself, in-process.

Measures the same trivial ASGI app with and without ``InferenceAdmission`` in front, for a
small non-streaming chat request and for a 200-chunk SSE stream.  No network, no engine:
this is the cost admitq adds on the event loop, nothing else.  Numbers depend on the
machine; run it on yours.

    python benchmarks/overhead.py
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from admitq import InferenceAdmission
from admitq.testing import call_asgi

CHAT = "/v1/chat/completions"
BODY = {
    "model": "m",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Summarise the plot of Hamlet in three sentences. " * 8},
    ],
    "max_tokens": 256,
}
JSON_REPLY = json.dumps(
    {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "x" * 800}}],
        "usage": {"prompt_tokens": 110, "completion_tokens": 200, "total_tokens": 310},
    }
).encode()
EVENT = b'data: {"id":"c","choices":[{"index":0,"delta":{"content":"tok "}}]}\n\n'


async def plain(scope: Any, receive: Any, send: Any) -> None:
    await receive()
    headers = [(b"content-type", b"application/json")]
    await send({"type": "http.response.start", "status": 200, "headers": headers})
    await send({"type": "http.response.body", "body": JSON_REPLY})


async def stream(scope: Any, receive: Any, send: Any) -> None:
    await receive()
    headers = [(b"content-type", b"text/event-stream")]
    await send({"type": "http.response.start", "status": 200, "headers": headers})
    for _ in range(200):
        await send({"type": "http.response.body", "body": EVENT, "more_body": True})
    await send({"type": "http.response.body", "body": b"data: [DONE]\n\n"})


async def measure(app: Any, count: int) -> float:
    for _ in range(200):  # warm-up
        await call_asgi(app, "POST", CHAT, json_body=BODY, keep_body=False)
    started = time.perf_counter()
    for _ in range(count):
        await call_asgi(app, "POST", CHAT, json_body=BODY, keep_body=False)
    return (time.perf_counter() - started) / count * 1e6


async def main() -> None:
    options: dict[str, Any] = {
        "max_inflight_tokens": 10**9,
        "max_inflight_requests": 10**6,
        "tenant_tpm": 10**12,
        "tenant_rpm": 10**9,
    }
    print(f"{'case':<28}{'bare app':>12}{'with admitq':>14}{'added':>10}   (microseconds/request)")
    for name, app, count in (
        ("JSON response", plain, 5000),
        ("SSE stream, 200 chunks", stream, 1000),
    ):
        bare = await measure(app, count)
        wrapped = await measure(InferenceAdmission(app, **options), count)
        print(f"{name:<28}{bare:>12.1f}{wrapped:>14.1f}{wrapped - bare:>10.1f}")


if __name__ == "__main__":
    asyncio.run(main())

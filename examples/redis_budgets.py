"""Tenant budgets shared across workers and replicas with Redis.

    pip install "admitq[redis] @ git+https://github.com/cb-Sigma/admitq" fastapi uvicorn
    REDIS_URL=redis://127.0.0.1:6379/0 uvicorn examples.redis_budgets:app --workers 4

With the default in-memory backend every worker process keeps its own budgets, so four
workers would let a tenant spend four times its TPM.  With ``RedisBackend`` all workers
debit the same buckets, atomically (one Lua script per request).

What stays per process, with either backend: the in-flight limits and the queue.  Divide
the engine's capacity by the number of workers, as below.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import FastAPI

from admitq import InferenceAdmission
from admitq.backends.redis import RedisBackend

WORKERS = int(os.environ.get("WEB_CONCURRENCY", "4"))
ENGINE_MAX_SEQS = 32  # what the model server can run at once (vLLM --max-num-seqs)
ENGINE_TOKENS = 64_000  # in-flight tokens you are comfortable with across all workers

app = FastAPI()
app.add_middleware(
    InferenceAdmission,
    backend=RedisBackend.from_url(os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")),
    # If Redis is slow or down: "open" admits without budget checks (the in-flight limits
    # still protect the engine), "closed" answers 503.
    on_backend_error="open",
    tenant_tpm={"default": 200_000, "free-tier": 20_000},
    tenant_rpm={"default": 600},
    # Per-process capacity: this worker's share of the engine.
    max_inflight_requests=max(1, ENGINE_MAX_SEQS // WORKERS),
    max_inflight_tokens=ENGINE_TOKENS // WORKERS,
    max_queue_s=2.0,
)


@app.post("/v1/chat/completions")
async def chat_completions(body: dict[str, Any]) -> dict[str, Any]:
    await asyncio.sleep(0.5)  # stand-in for the call to your model server
    return {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 1},
    }

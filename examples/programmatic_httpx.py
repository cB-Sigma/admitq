"""Using admitq without ASGI: admission control inside a plain httpx client or worker.

Useful when there is no web framework in the picture - a batch worker pulling jobs from a
queue, a CLI, an agent loop - but many coroutines still share one model server.

    pip install "admitq @ git+https://github.com/cb-Sigma/admitq" httpx
    UPSTREAM=http://127.0.0.1:8000 python examples/programmatic_httpx.py
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from admitq import AdmissionController, Shed

UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:8000")

controller = AdmissionController(
    max_inflight_tokens=16_000,
    max_inflight_requests=8,
    max_queue_s=30.0,  # a worker can afford to wait longer than a web request
    tenant_tpm={"default": 120_000},
    priorities={"interactive": 0, "batch": 1},
)


async def complete(
    client: httpx.AsyncClient, tenant: str, body: dict[str, Any], *, priority: str = "batch"
) -> dict[str, Any] | None:
    """One chat completion, admitted through the controller.  Returns ``None`` if shed."""
    cost = controller.estimate(body, "chat")
    try:
        async with controller.admit(tenant=tenant, cost=cost, priority=priority) as ticket:
            response = await client.post("/v1/chat/completions", json=body)
            ticket.first_byte()
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            # Correct the estimate with what the engine actually counted.
            ticket.report_usage(**payload.get("usage", {}))
            return payload
    except Shed as shed:
        # shed.status is the HTTP status a server would answer with; shed.retry_after is
        # how long to back off (exact for budgets, estimated for overload).
        print(f"[{tenant}] shed: {shed.reason}, retry in {shed.retry_after:.1f}s")
        return None


async def main() -> None:
    prompts = [f"Give me fact number {i} about the ocean." for i in range(40)]
    async with httpx.AsyncClient(base_url=UPSTREAM, timeout=120.0) as client:
        results = await asyncio.gather(
            *(
                complete(
                    client,
                    tenant=f"team-{i % 3}",
                    body={
                        "model": "default",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 64,
                    },
                )
                for i, prompt in enumerate(prompts)
            )
        )
    done = sum(result is not None for result in results)
    print(f"{done}/{len(prompts)} completed")
    print(controller.metrics.render())
    await controller.aclose()


if __name__ == "__main__":
    asyncio.run(main())

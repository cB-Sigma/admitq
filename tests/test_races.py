"""Concurrency stress: many overlapping requests with disconnects and cancellations.

Each seed is a different interleaving; all of them run on virtual time, so thousands of
simulated seconds take milliseconds and every failure is reproducible from its seed.
"""

from __future__ import annotations

import asyncio
import random

import pytest

from admitq import InferenceAdmission
from admitq.backends import MemoryBackend
from admitq.testing import FakeModelBackend, call_asgi, run_virtual

from .helpers import tenant_header

CHAT = "/v1/chat/completions"


class Probe:
    """Sits between admitq and the backend and checks the limits on every entry."""

    def __init__(self, app, limit_requests: int, limit_tokens: int) -> None:
        self.app = app
        self.limit_requests = limit_requests
        self.limit_tokens = limit_tokens
        self.concurrent = 0
        self.peak = 0
        self.calls = 0
        self.controller = None

    async def __call__(self, scope, receive, send):
        self.calls += 1
        self.concurrent += 1
        self.peak = max(self.peak, self.concurrent)
        try:
            assert self.concurrent <= self.limit_requests
            snapshot = self.controller.snapshot()
            assert snapshot.inflight_requests <= self.limit_requests
            assert snapshot.inflight_tokens <= self.limit_tokens
            # Everything the app is running is accounted for by the controller.
            assert snapshot.inflight_requests >= self.concurrent
            await self.app(scope, receive, send)
        finally:
            self.concurrent -= 1


class SlowBackend:
    """A budget backend with latency, so reservations interleave with everything else."""

    def __init__(self, clock, rng: random.Random) -> None:
        self.inner = MemoryBackend(clock=clock)
        self.rng = rng

    async def reserve(self, tenant, tokens, requests, limits):
        await asyncio.sleep(self.rng.uniform(0, 0.02))
        return self.inner.reserve_sync(tenant, tokens, requests, limits)

    async def adjust(self, tenant, tokens, requests, limits):
        await asyncio.sleep(self.rng.uniform(0, 0.02))
        self.inner.adjust_sync(tenant, tokens, requests, limits)

    async def aclose(self):
        pass


async def storm(seed: int, *, slow_backend: bool, propagate: bool) -> None:
    rng = random.Random(seed)
    loop = asyncio.get_running_loop()
    limit_requests, limit_tokens = rng.randint(1, 6), rng.randint(300, 3000)
    backend = FakeModelBackend(
        max_num_seqs=rng.randint(1, 8),
        total_tps=rng.choice([50, 200, 1000]),
        stream_tps=rng.choice([10, 50]),
        abort_on_disconnect=rng.random() < 0.7,
    )
    probe = Probe(backend, limit_requests, limit_tokens)
    options = {}
    if slow_backend:
        options["backend"] = SlowBackend(loop.time, rng)
    app = InferenceAdmission(
        probe,
        max_inflight_requests=limit_requests,
        max_inflight_tokens=limit_tokens,
        max_queue_s=rng.choice([0.0, 0.5, 3.0]),
        max_queue_depth=rng.choice([None, 0, 5, 50]),
        max_tenant_share=rng.choice([None, 0.5]),
        tenant_tpm=rng.choice([None, 20_000, 200_000]),
        tenant_rpm=rng.choice([None, 600]),
        priorities={"interactive": 0, "batch": 1},
        tenant_key=tenant_header,
        cancel_on_disconnect=propagate,
        early_shed=rng.random() < 0.5,
        **options,
    )
    probe.controller = app.controller
    outcomes: dict[str, int] = {}

    async def client(index: int) -> None:
        await asyncio.sleep(rng.uniform(0, 20))
        body = {
            "messages": [{"role": "user", "content": "x" * rng.randint(0, 2000)}],
            "max_tokens": rng.randint(1, 300),
            "stream": rng.random() < 0.7,
        }
        if rng.random() < 0.5:
            body["stream_options"] = {"include_usage": True}
        headers = {
            "x-tenant-id": rng.choice(["a", "b", "c"]),
            "x-priority": rng.choice(["interactive", "batch"]),
        }
        if rng.random() < 0.2:
            headers["x-request-timeout"] = str(rng.uniform(0.01, 5))
        behaviour = rng.choice(["normal", "normal", "hang-up-chunks", "hang-up-time", "cancel"])
        kwargs = {}
        if behaviour == "hang-up-chunks":
            kwargs["disconnect_after_chunks"] = rng.randint(1, 20)
        elif behaviour == "hang-up-time":
            kwargs["disconnect_after_s"] = rng.uniform(0, 6)
        call = call_asgi(
            app, "POST", CHAT, json_body=body, headers=headers, keep_body=False, **kwargs
        )
        if behaviour == "cancel":
            task = asyncio.ensure_future(call)
            await asyncio.sleep(rng.uniform(0, 6))
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                outcomes["cancelled"] = outcomes.get("cancelled", 0) + 1
                return
            response = task.result()
        else:
            response = await call
        key = "disconnected" if response.disconnected else str(response.status)
        outcomes[key] = outcomes.get(key, 0) + 1

    await asyncio.gather(*(client(index) for index in range(300)))
    await asyncio.sleep(1.0)  # let background settlements finish

    controller = app.controller
    snapshot = controller.snapshot()
    context = f"seed={seed} outcomes={outcomes}"
    # Nothing leaked, anywhere.
    assert snapshot.inflight_requests == 0, context
    assert snapshot.inflight_tokens == 0, context
    assert snapshot.queued_requests == 0, context
    assert snapshot.queued_tokens == 0, context
    assert probe.concurrent == 0, context
    assert backend.running == 0, context
    assert backend.waiting == 0, context
    assert not controller.scheduler._tenant_inflight, context
    assert not controller._tasks, context
    # Every admitted request was released exactly once.
    metrics = controller.metrics
    admitted = metrics.total("admitq_admitted_total")
    completed = metrics.total("admitq_completed_total") - metrics.total(
        "admitq_completed_total", outcome="abandoned_in_queue"
    )
    assert admitted == completed, context
    # Requests cancelled between admission and the app call never reach the app.
    assert probe.calls <= admitted, context
    assert probe.peak <= limit_requests, context
    # The storm did exercise the interesting paths.
    assert admitted > 0, context


@pytest.mark.parametrize("seed", range(40))
def test_storm_in_memory(seed):
    run_virtual(storm(seed, slow_backend=False, propagate=seed % 2 == 0))


@pytest.mark.parametrize("seed", range(100, 120))
def test_storm_with_async_budget_backend(seed):
    run_virtual(storm(seed, slow_backend=True, propagate=seed % 2 == 1))


def test_storms_cover_every_outcome():
    """Guard against the stress test silently degenerating into the happy path."""
    seen: set[str] = set()

    async def one(seed: int) -> None:
        rng = random.Random(seed)
        backend = FakeModelBackend(max_num_seqs=2, total_tps=100, stream_tps=20)
        app = InferenceAdmission(
            backend,
            max_inflight_requests=2,
            max_queue_s=1.0,
            tenant_tpm=1_500,
            tenant_key=tenant_header,
        )

        async def client() -> None:
            await asyncio.sleep(rng.uniform(0, 10))
            body = {"messages": [], "max_tokens": rng.randint(1, 100), "stream": True}
            kwargs = {"disconnect_after_chunks": 3} if rng.random() < 0.3 else {}
            await call_asgi(app, "POST", CHAT, json_body=body, **kwargs)

        await asyncio.gather(*(client() for _ in range(200)))
        metrics = app.controller.metrics
        for reason in ("queue_timeout", "predicted_wait", "tenant_tokens"):
            if metrics.total("admitq_shed_total", reason=reason):
                seen.add(reason)
        for outcome in ("ok", "client_disconnect"):
            if metrics.total("admitq_completed_total", outcome=outcome):
                seen.add(outcome)

    for seed in range(5):
        run_virtual(one(seed))
    assert seen == {"queue_timeout", "predicted_wait", "tenant_tokens", "ok", "client_disconnect"}

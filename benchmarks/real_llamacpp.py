"""REAL-engine overload experiment: llama.cpp (via llama-cpp-python's server) on CPU.

What is real here: the model server, the model, the HTTP hops, the FastAPI proxy from
``examples/fastapi_proxy.py`` and the admitq middleware inside it.  What is small: one tiny
model (SmolLM2-135M-Instruct, 4-bit) on a few CPU cores, one generation at a time.  This is
*not* a GPU engine with continuous batching, and the numbers do not transfer to one.  It
shows the mechanism on a real engine; nothing more.

Topology (all on localhost)::

    load generator --HTTP--> uvicorn + FastAPI proxy [+ admitq] --HTTP--> llama_cpp.server

Usage::

    python benchmarks/real_llamacpp.py \
        --server-python /path/to/venv-with-llama-cpp-python/bin/python \
        --model /path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf

Needs ``fastapi``, ``uvicorn`` and ``httpx`` in the interpreter that runs this script, and
``llama-cpp-python[server]`` in the one given by ``--server-python`` (it may be the same).

The server is started with ``--interrupt_requests false``: with the default (``true``)
llama-cpp-python aborts the running generation whenever another request arrives, which is
a different failure mode from the unbounded queue studied here.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import json
import random
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import uvicorn

from admitq.testing import LoadReport, RequestRecord, TenantLoad

ROOT = Path(__file__).resolve().parent.parent
CHAT = "/v1/chat/completions"

PROMPTS = [
    "Explain what a hash table is.",
    "Write a short story about a lighthouse keeper.",
    "List reasons why the sky looks blue.",
    "Describe how bread is made.",
    "What is the difference between a list and a tuple in Python?",
]


def load_proxy_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "fastapi_proxy", ROOT / "examples" / "fastapi_proxy.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def llama_server(python: str, model: str, threads: int) -> Any:
    port = free_port()
    command = [
        python, "-m", "llama_cpp.server",
        "--model", model,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--n_ctx", "2048",
        "--n_threads", str(threads),
        "--interrupt_requests", "false",
    ]  # fmt: skip
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("llama_cpp.server exited during start-up")
            try:
                if httpx.get(f"{url}/v1/models", timeout=2.0).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        else:
            raise RuntimeError("llama_cpp.server did not become ready")
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()


@contextlib.asynccontextmanager
async def serve(app: Any) -> AsyncIterator[str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        await asyncio.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await task


async def calibrate(upstream: str) -> float:
    """Tokens per second of one uncontended generation (after a warm-up)."""
    body = {"messages": [{"role": "user", "content": PROMPTS[1]}], "max_tokens": 96}
    async with httpx.AsyncClient(base_url=upstream, timeout=120.0) as client:
        await client.post(CHAT, json={**body, "max_tokens": 8})
        started = time.perf_counter()
        response = await client.post(CHAT, json=body)
        elapsed = time.perf_counter() - started
    return float(response.json()["usage"]["completion_tokens"]) / elapsed


async def drive(
    base: str,
    tenants: list[TenantLoad],
    *,
    duration_s: float,
    seed: int,
    slo_s: float,
    client_timeout_s: float,
) -> LoadReport:
    """Open-loop Poisson load over real HTTP."""
    records: list[RequestRecord] = []
    tasks: list[asyncio.Task[None]] = []
    origin = time.perf_counter()
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=0)

    async with httpx.AsyncClient(base_url=base, limits=limits) as client:

        async def one(load: TenantLoad, prompt: str) -> None:
            sent_at = time.perf_counter() - origin
            body = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": load.max_tokens,
                "stream": True,
                "temperature": 0.7,
            }
            status = 0
            ttfb: float | None = None
            tokens = 0
            outcome = "error"
            reason: str | None = None
            queue_s = 0.0

            async def consume() -> None:
                nonlocal status, ttfb, tokens, reason, queue_s
                async with client.stream(
                    "POST", CHAT, json=body, headers={"x-tenant-id": load.name}, timeout=None
                ) as response:
                    status = response.status_code
                    reason = response.headers.get("x-admitq-shed-reason")
                    queue_s = float(response.headers.get("x-admitq-queue-ms", "0")) / 1000
                    async for line in response.aiter_lines():
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        delta = json.loads(line[6:])["choices"][0].get("delta", {})
                        if delta.get("content"):
                            if ttfb is None:
                                ttfb = time.perf_counter() - origin - sent_at
                            tokens += 1

            try:
                await asyncio.wait_for(consume(), client_timeout_s)
                if status == 200:
                    outcome = "ok"
                elif status in (429, 503):
                    outcome = "shed"
            except asyncio.TimeoutError:
                outcome = "timeout"
            except (httpx.HTTPError, ValueError, KeyError, IndexError):
                outcome = "error"
            records.append(
                RequestRecord(
                    tenant=load.name,
                    sent_at=sent_at,
                    status=status,
                    outcome=outcome,
                    latency=time.perf_counter() - origin - sent_at,
                    ttfb=ttfb,
                    queue_s=queue_s,
                    shed_reason=reason,
                    output_tokens=tokens if outcome == "ok" else 0,
                )
            )

        async def arrivals(load: TenantLoad, rng: random.Random) -> None:
            at = rng.expovariate(load.rps)
            while at < duration_s:
                await asyncio.sleep(max(0.0, origin + at - time.perf_counter()))
                tasks.append(asyncio.create_task(one(load, rng.choice(PROMPTS))))
                at += rng.expovariate(load.rps)

        await asyncio.gather(
            *(arrivals(load, random.Random(f"{seed}:{load.name}")) for load in tenants)
        )
        await asyncio.gather(*tasks)
    records.sort(key=lambda record: record.sent_at)
    return LoadReport(duration_s, slo_s, records, [load.name for load in tenants])


async def run(args: argparse.Namespace, upstream: str) -> None:
    proxy = load_proxy_module()
    rate = await calibrate(upstream)
    print(f"calibration: {rate:.0f} tokens/s for a single uncontended request\n")

    tenants = [
        TenantLoad("alice", rps=args.quiet_rps, max_tokens=args.quiet_tokens),
        TenantLoad("bob", rps=args.quiet_rps, max_tokens=args.quiet_tokens),
        TenantLoad("noisy", rps=args.noisy_rps, max_tokens=args.noisy_tokens),
    ]
    demand = sum(load.rps * load.max_tokens for load in tenants)
    print(
        f"offered load: up to {demand:.0f} output tokens/s against ~{rate:.0f} "
        f"({demand / rate:.1f}x), {args.duration:.0f} s per arm, seed {args.seed}"
    )
    print(f"good = completed within {args.slo:.0f} s; clients give up after {args.timeout:.0f} s\n")

    admission = {
        # The engine generates one request at a time: keep one running and one ready.
        "max_inflight_requests": 2,
        # Room for one noisy request plus quiet ones; with the share cap below, the noisy
        # tenant can never hold both in-flight places.
        "max_inflight_tokens": 2 * (args.noisy_tokens + 60),
        "max_tenant_share": 0.5,
        # On a one-at-a-time engine the queue bound has to exceed the longest job.
        "max_queue_s": args.max_queue,
        "cancel_on_disconnect": True,
        "tenant_key": lambda info: info.headers.get("x-tenant-id", "anonymous"),
        "metrics_path": "/admitq/metrics",
    }
    arms: dict[str, dict[str, Any] | None] = {"no admission control": None, "admitq": admission}
    for name, config in arms.items():
        if args.arm not in ("both", name.split()[0]):
            continue
        app = proxy.create_app(upstream, admission=config)
        async with serve(app) as base:
            report = await drive(
                base,
                tenants,
                duration_s=args.duration,
                seed=args.seed,
                slo_s=args.slo,
                client_timeout_s=args.timeout,
            )
            print(f"== {name}")
            print(report.format())
            if config is not None:
                async with httpx.AsyncClient() as scraper:
                    metrics = (await scraper.get(f"{base}/admitq/metrics")).text
                wanted = (
                    "admitq_shed_total{",
                    "admitq_actual_tokens_total{",
                    "admitq_completed_total{",
                )
                for line in metrics.splitlines():
                    if line.startswith(wanted):
                        print("   ", line)
            print()
        await asyncio.sleep(3.0)  # let the engine drain anything left over


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--server-python", default=sys.executable)
    parser.add_argument("--model", required=True)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slo", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--quiet-rps", type=float, default=0.25)
    parser.add_argument("--quiet-tokens", type=int, default=16)
    parser.add_argument("--noisy-rps", type=float, default=0.8)
    parser.add_argument("--noisy-tokens", type=int, default=96)
    parser.add_argument("--max-queue", type=float, default=6.0)
    parser.add_argument("--arm", choices=["both", "no", "admitq"], default="both")
    args = parser.parse_args()
    with llama_server(args.server_python, args.model, args.threads) as upstream:
        asyncio.run(run(args, upstream))


if __name__ == "__main__":
    main()

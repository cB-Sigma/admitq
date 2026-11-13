"""Reading congestion signals from engine metrics endpoints.

admitq has no HTTP client of its own.  These helpers parse the Prometheus text an engine
serves, and :func:`poll_signal` turns "a coroutine that fetches that text" into a steady
feed for the adaptive limit::

    async def fetch() -> str:
        return (await client.get("http://vllm:8000/metrics")).text

    task = asyncio.create_task(
        poll_signal(controller, fetch, parse=vllm_requests_waiting, interval_s=1.0)
    )

Metric names verified against the engines' sources:

* vLLM: ``vllm:num_requests_waiting`` and ``vllm:num_requests_running`` (gauges).
* llama.cpp server (started with ``--metrics``): ``llamacpp:requests_deferred`` and
  ``llamacpp:requests_processing`` (gauges).
* SGLang: ``sglang:num_queue_reqs`` and ``sglang:num_running_reqs`` (gauges).
* TGI: ``tgi_queue_size`` (gauge).

A scraped gauge is always somewhat stale; treat it as a trend, not as a live count.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .controller import AdmissionController

__all__ = [
    "llamacpp_requests_deferred",
    "poll_signal",
    "prometheus_value",
    "sglang_queue_requests",
    "tgi_queue_size",
    "vllm_requests_waiting",
]

logger = logging.getLogger("admitq")


def prometheus_value(text: str, name: str) -> float | None:
    """Sum of all samples of metric ``name`` in Prometheus text exposition, or ``None``
    if the metric is absent.  Labels are ignored (samples are summed across them)."""
    total: float | None = None
    for line in text.splitlines():
        if not line.startswith(name):
            continue
        rest = line[len(name) :]
        if not rest or rest[0] not in " {\t":
            continue  # a longer metric name sharing the prefix
        if rest[0] == "{":
            end = rest.rfind("}")
            if end < 0:
                continue
            rest = rest[end + 1 :]
        fields = rest.split()
        if not fields:
            continue
        try:
            value = float(fields[0])
        except ValueError:
            continue
        total = value if total is None else total + value
    return total


def vllm_requests_waiting(text: str) -> float | None:
    return prometheus_value(text, "vllm:num_requests_waiting")


def llamacpp_requests_deferred(text: str) -> float | None:
    return prometheus_value(text, "llamacpp:requests_deferred")


def sglang_queue_requests(text: str) -> float | None:
    return prometheus_value(text, "sglang:num_queue_reqs")


def tgi_queue_size(text: str) -> float | None:
    return prometheus_value(text, "tgi_queue_size")


async def poll_signal(
    controller: AdmissionController,
    fetch: Callable[[], Awaitable[str]],
    *,
    parse: Callable[[str], float | None] = vllm_requests_waiting,
    interval_s: float = 1.0,
) -> None:
    """Poll an engine metric forever and feed it to ``controller.report_signal``.

    Run it as a task and cancel it on shutdown.  Fetch or parse failures are logged and
    skipped: no sample is better than a made-up one.
    """
    while True:
        try:
            value = parse(await fetch())
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("admitq: engine signal poll failed", exc_info=True)
            value = None
        if value is not None:
            controller.report_signal(value)
        await asyncio.sleep(interval_s)

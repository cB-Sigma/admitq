"""Helpers for testing and tuning admission settings.  Standard library only.

* :class:`FakeClock` - a clock you advance by hand, for driving the scheduler directly.
* :func:`run_virtual` / :class:`VirtualTimeLoop` - an event loop whose clock jumps to the
  next timer instead of sleeping, so a ten-minute load test runs in a second and gives the
  same answer every time.
* :class:`FakeModelBackend` - an ASGI app that behaves like an inference server with
  continuous batching and an unbounded internal queue (OpenAI-style routes, SSE streaming,
  ``usage`` reporting).
* :func:`call_asgi` - call any ASGI app without an HTTP library; can disconnect mid-stream.
* :func:`simulate_load` - open-loop load with several tenants; returns latency
  percentiles, shed rate and per-tenant goodput.

Example - "does my configuration protect the quiet tenant?"::

    from admitq import InferenceAdmission
    from admitq.testing import FakeModelBackend, TenantLoad, run_virtual, simulate_load

    def test_quiet_tenant_is_protected():
        app = InferenceAdmission(
            FakeModelBackend(max_num_seqs=8, total_tps=400),
            max_inflight_tokens=8_000, max_queue_s=2.0, tenant_tpm=60_000,
            tenant_key=lambda info: info.headers.get("x-tenant-id", "anonymous"),
        )
        report = run_virtual(simulate_load(
            app,
            [TenantLoad("quiet", rps=1), TenantLoad("noisy", rps=30)],
            duration_s=120, slo_s=10,
        ))
        assert report.tenant("quiet").goodput_fraction > 0.95
"""

from __future__ import annotations

import asyncio
import json
import math
import random
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

from .usage import ResponseObserver

__all__ = [
    "ASGIResponse",
    "FakeClock",
    "FakeModelBackend",
    "LoadReport",
    "RequestRecord",
    "TenantLoad",
    "TenantStats",
    "VirtualTimeLoop",
    "call_asgi",
    "percentile",
    "run_virtual",
    "simulate_load",
]

T = TypeVar("T")
Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


# --------------------------------------------------------------------------- clocks


class FakeClock:
    """A manually advanced clock.  Callable, so it can be passed as ``clock=``."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("time only moves forward")
        self.now += seconds
        return self.now


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    """An asyncio event loop with virtual time.

    Whenever the loop would sleep until its next timer, the clock jumps there instead.
    ``asyncio.sleep(60)`` returns immediately and ``loop.time()`` reads 60 seconds later.
    Everything driven by the loop clock - including admitq's queue deadlines and token
    buckets - behaves exactly as it would in real time, deterministically.

    Only for code that stays inside the process: real sockets, subprocesses and threads
    live in real time and will not line up with the virtual clock.
    """

    def __init__(self) -> None:
        super().__init__()
        self._virtual_now = 0.0
        self._idle_rounds = 0
        selector: Any = getattr(self, "_selector")  # noqa: B009 - private, but stable
        real_select = selector.select

        def select(timeout: float | None = None) -> Any:
            if timeout is None:
                # Nothing scheduled.  Give threads a moment to call back, then give up
                # rather than hang the test run forever.
                self._idle_rounds += 1
                if self._idle_rounds > 100:
                    raise RuntimeError(
                        "VirtualTimeLoop is idle with nothing scheduled: the code under "
                        "test is waiting for something that will never happen"
                    )
                return real_select(0.05)
            self._idle_rounds = 0
            if timeout > 0:
                self._virtual_now += timeout
            return real_select(0)

        selector.select = select

    def time(self) -> float:
        return self._virtual_now


def run_virtual(main: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine to completion on a fresh :class:`VirtualTimeLoop`."""
    loop = VirtualTimeLoop()
    try:
        return loop.run_until_complete(main)
    finally:
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()


# --------------------------------------------------------------------------- fake engine


class FakeModelBackend:
    """An ASGI app that imitates a self-hosted inference server under load.

    The model is deliberately simple and captures what matters for admission control:

    * up to ``max_num_seqs`` requests decode concurrently (continuous batching);
    * the engine produces at most ``total_tps`` tokens per second in total and at most
      ``stream_tps`` per request, so each request slows down as the batch fills;
    * prompts are processed at ``prefill_tps`` tokens per second before the first token;
    * anything beyond ``max_num_seqs`` waits in an internal FIFO queue that is **unbounded**
      (``max_waiting=None``) - the behaviour that makes an overloaded engine slow for
      everyone - or is refused with 503 once ``max_waiting`` requests are waiting.

    Routes: ``POST /v1/chat/completions``, ``/v1/completions``, ``/v1/responses`` and
    ``/v1/embeddings``, ``GET /metrics`` (vLLM-style gauges) and ``GET /health``.  Any
    path prefix is accepted.

    Request extensions (ignored by real servers): ``x_fake_output_tokens`` sets how many
    tokens are generated (default: ``max_tokens``, or ``default_output_tokens``);
    ``x_fake_prompt_tokens`` overrides the prompt size.

    Every generated token is the four characters ``"tok "``, so the default character
    heuristic counts streamed output exactly.
    """

    def __init__(
        self,
        *,
        max_num_seqs: int = 8,
        total_tps: float = 400.0,
        stream_tps: float = 50.0,
        prefill_tps: float = 4000.0,
        max_waiting: int | None = None,
        default_output_tokens: int = 64,
        abort_on_disconnect: bool = True,
        report_usage: bool = True,
        model: str = "fake-model",
    ) -> None:
        self.max_num_seqs = max_num_seqs
        self.total_tps = total_tps
        self.stream_tps = stream_tps
        self.prefill_tps = prefill_tps
        self.max_waiting = max_waiting
        self.default_output_tokens = default_output_tokens
        self.abort_on_disconnect = abort_on_disconnect
        self.report_usage = report_usage
        self.model = model
        self._waiters: deque[asyncio.Future[None]] = deque()
        # Observable state.
        self.running = 0
        self.waiting = 0
        self.peak_running = 0
        self.peak_waiting = 0
        self.received = 0
        self.completed = 0
        self.aborted = 0
        self.rejected = 0
        self.generated_tokens = 0
        self.prompt_tokens = 0
        #: Parsed JSON bodies of the inference requests received, most recent last.
        self.bodies: deque[Any] = deque(maxlen=64)

    # -- ASGI -------------------------------------------------------------------------

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        path: str = scope["path"].rstrip("/")
        method: str = scope["method"]
        if method == "GET" and path.endswith("/metrics"):
            await _respond(send, 200, self.metrics_text().encode(), b"text/plain; version=0.0.4")
        elif method == "GET" and path.endswith("/health"):
            await _respond(send, 200, b'{"status":"ok"}')
        elif method == "POST" and path.endswith("/embeddings"):
            await self._embeddings(receive, send)
        elif method == "POST" and path.endswith(
            ("/chat/completions", "/completions", "/responses")
        ):
            await self._generate(path, receive, send)
        else:
            await _respond(send, 404, b'{"error":{"message":"not found"}}')

    def metrics_text(self) -> str:
        """Prometheus text with the two gauges vLLM exposes for queue state."""
        label = f'{{model_name="{self.model}"}}'
        return (
            "# TYPE vllm:num_requests_running gauge\n"
            f"vllm:num_requests_running{label} {self.running}\n"
            "# TYPE vllm:num_requests_waiting gauge\n"
            f"vllm:num_requests_waiting{label} {self.waiting}\n"
        )

    # -- internals --------------------------------------------------------------------

    async def _read_json(self, receive: Receive) -> Any:
        chunks: list[bytes] = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return None
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        try:
            return json.loads(b"".join(chunks))
        except ValueError:
            return ...

    def _wake(self) -> None:
        while self._waiters and self.running < self.max_num_seqs:
            waiter = self._waiters.popleft()
            if not waiter.done():
                self.running += 1
                self.peak_running = max(self.peak_running, self.running)
                waiter.set_result(None)

    def _prompt_tokens(self, body: Mapping[str, Any]) -> int:
        override = body.get("x_fake_prompt_tokens")
        if isinstance(override, int):
            return override
        chars = 0
        messages = body.get("messages")
        if isinstance(messages, list):
            for message in messages:
                content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, str):
                    chars += len(content)
        for key in ("prompt", "input"):
            value = body.get(key)
            if isinstance(value, str):
                chars += len(value)
            elif isinstance(value, list):
                chars += sum(len(item) for item in value if isinstance(item, str))
        return math.ceil(chars / 4)

    async def _embeddings(self, receive: Receive, send: Send) -> None:
        body = await self._read_json(receive)
        if body is None:
            return
        if not isinstance(body, dict):
            await _respond(send, 400, b'{"error":{"message":"invalid JSON body"}}')
            return
        self.received += 1
        self.bodies.append(body)
        prompt_tokens = self._prompt_tokens(body)
        await asyncio.sleep(prompt_tokens / self.prefill_tps)
        raw = body.get("input")
        count = len(raw) if isinstance(raw, list) else 1
        payload = {
            "object": "list",
            "model": self.model,
            "data": [
                {"object": "embedding", "index": i, "embedding": [0.0] * 8} for i in range(count)
            ],
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
        }
        self.completed += 1
        self.prompt_tokens += prompt_tokens
        await _respond(send, 200, json.dumps(payload).encode())

    async def _generate(self, path: str, receive: Receive, send: Send) -> None:
        body = await self._read_json(receive)
        if body is None:
            return
        if not isinstance(body, dict):
            await _respond(send, 400, b'{"error":{"message":"invalid JSON body"}}')
            return
        self.received += 1
        self.bodies.append(body)
        chat = path.endswith("/chat/completions")
        stream = body.get("stream") is True
        include_usage = False
        stream_options = body.get("stream_options")
        if isinstance(stream_options, dict):
            include_usage = bool(stream_options.get("include_usage"))
        requested = None
        for key in (
            "x_fake_output_tokens",
            "max_completion_tokens",
            "max_tokens",
            "max_output_tokens",
        ):
            if isinstance(body.get(key), int):
                requested = int(body[key])
                break
        output_tokens = self.default_output_tokens if requested is None else max(0, requested)
        prompt_tokens = self._prompt_tokens(body)

        must_wait = self.running >= self.max_num_seqs or bool(self._waiters)
        if must_wait and self.max_waiting is not None and self.waiting >= self.max_waiting:
            self.rejected += 1
            await _respond(send, 503, b'{"error":{"message":"engine queue is full"}}')
            return

        loop = asyncio.get_running_loop()
        disconnected = False
        waiter: asyncio.Future[None] | None = None

        async def watch() -> None:
            nonlocal disconnected
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    disconnected = True
                    if self.abort_on_disconnect and waiter is not None and not waiter.done():
                        waiter.cancel()
                    return

        watcher = loop.create_task(watch())
        holding = False
        generated = 0
        try:
            # Engine admission: a free batch slot, or the unbounded waiting queue.
            if self.running < self.max_num_seqs and not self._waiters:
                self.running += 1
                self.peak_running = max(self.peak_running, self.running)
            else:
                waiter = loop.create_future()
                self._waiters.append(waiter)
                self.waiting += 1
                self.peak_waiting = max(self.peak_waiting, self.waiting)
                try:
                    await waiter
                except asyncio.CancelledError:
                    if waiter.done() and not waiter.cancelled():
                        # A slot was granted in the same instant: hand it on.
                        self.running -= 1
                        self._wake()
                    if disconnected and self.abort_on_disconnect:
                        self.aborted += 1
                        return
                    raise
                finally:
                    self.waiting -= 1
            holding = True

            def gone() -> bool:
                return disconnected and self.abort_on_disconnect

            await asyncio.sleep(prompt_tokens / self.prefill_tps)
            self.prompt_tokens += prompt_tokens
            if gone():
                self.aborted += 1
                return

            created = 0
            request_id = f"fake-{self.received}"
            if stream:
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [
                            (b"content-type", b"text/event-stream; charset=utf-8"),
                            (b"cache-control", b"no-cache"),
                        ],
                    }
                )
            for _ in range(output_tokens):
                rate = min(self.stream_tps, self.total_tps / max(1, self.running))
                await asyncio.sleep(1.0 / rate)
                if gone():
                    self.aborted += 1
                    return
                generated += 1
                self.generated_tokens += 1
                if stream:
                    if chat:
                        choice: dict[str, Any] = {
                            "index": 0,
                            "delta": {"content": "tok "},
                            "finish_reason": None,
                        }
                        obj = "chat.completion.chunk"
                    else:
                        choice = {"index": 0, "text": "tok ", "finish_reason": None}
                        obj = "text_completion"
                    event = {
                        "id": request_id,
                        "object": obj,
                        "created": created,
                        "model": self.model,
                        "choices": [choice],
                    }
                    await send(_sse(event, more=True))
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": generated,
                "total_tokens": prompt_tokens + generated,
            }
            if stream:
                if include_usage and self.report_usage:
                    final: dict[str, Any] = {
                        "id": request_id,
                        "object": "chat.completion.chunk" if chat else "text_completion",
                        "created": created,
                        "model": self.model,
                        "choices": [],
                        "usage": usage,
                    }
                    await send(_sse(final, more=True))
                await send(
                    {"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": False}
                )
            else:
                text = "tok " * generated
                if chat:
                    choice = {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "length",
                    }
                else:
                    choice = {"index": 0, "text": text, "finish_reason": "length"}
                payload: dict[str, Any] = {
                    "id": request_id,
                    "object": "chat.completion" if chat else "text_completion",
                    "created": created,
                    "model": self.model,
                    "choices": [choice],
                }
                if self.report_usage:
                    payload["usage"] = usage
                await _respond(send, 200, json.dumps(payload).encode())
            self.completed += 1
        finally:
            if holding:
                self.running -= 1
                self._wake()
            watcher.cancel()


def _sse(event: Mapping[str, Any], *, more: bool) -> Message:
    data = b"data: " + json.dumps(event, separators=(",", ":")).encode() + b"\n\n"
    return {"type": "http.response.body", "body": data, "more_body": more}


async def _respond(
    send: Send, status: int, body: bytes, content_type: bytes = b"application/json"
) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", content_type),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


# --------------------------------------------------------------------------- ASGI client


@dataclass(slots=True)
class ASGIResponse:
    """Result of :func:`call_asgi`."""

    status: int = 0
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    #: Number of non-empty body chunks received.
    chunks: int = 0
    #: Seconds from the call to the first body byte (``None`` if there was none).
    ttfb: float | None = None
    #: Seconds from the call until the application returned.
    duration: float = 0.0
    #: The application sent a complete response.
    completed: bool = False
    #: The client disconnected before the response completed.
    disconnected: bool = False

    def json(self) -> Any:
        return json.loads(self.body)

    def sse_events(self) -> list[Any]:
        """Decoded ``data:`` payloads of a server-sent-events body (``[DONE]`` dropped)."""
        events: list[Any] = []
        for line in self.body.split(b"\n"):
            line = line.strip()
            if line.startswith(b"data:"):
                data = line[5:].strip()
                if data and data != b"[DONE]":
                    events.append(json.loads(data))
        return events


async def call_asgi(
    app: ASGIApp,
    method: str,
    path: str,
    *,
    json_body: Any = None,
    body: bytes | Iterable[bytes] | None = None,
    headers: Mapping[str, str] | None = None,
    disconnect_after_chunks: int | None = None,
    disconnect_after_s: float | None = None,
    keep_body: bool = True,
) -> ASGIResponse:
    """Call an ASGI app in-process, without a server or an HTTP client.

    Args:
        json_body: JSON-encoded as the request body (sets ``content-type``).
        body: raw body bytes, or an iterable of chunks to deliver one ``http.request``
            message at a time.
        disconnect_after_chunks: simulate the client going away after this many response
            body chunks (0 = before the response starts is not possible; use
            ``disconnect_after_s``).
        disconnect_after_s: simulate the client going away after this many seconds
            (a client-side timeout).
        keep_body: set to ``False`` to drop body bytes (for long streams in load tests).

    As with a real server, sends after a disconnect are silently dropped and the next
    ``receive()`` returns ``http.disconnect``.
    """
    loop = asyncio.get_running_loop()
    request_headers = {k.lower(): v for k, v in (headers or {}).items()}
    if json_body is not None:
        payloads = [json.dumps(json_body).encode()]
        request_headers.setdefault("content-type", "application/json")
    elif body is None:
        payloads = [b""]
    elif isinstance(body, bytes | bytearray):
        payloads = [bytes(body)]
    else:
        payloads = list(body) or [b""]
    if "content-length" not in request_headers and len(payloads) == 1:
        request_headers["content-length"] = str(len(payloads[0]))
    raw_path, _, query = path.partition("?")
    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": raw_path,
        "raw_path": raw_path.encode(),
        "query_string": query.encode(),
        "root_path": "",
        "headers": [(k.encode("latin-1"), v.encode("latin-1")) for k, v in request_headers.items()],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }
    response = ASGIResponse()
    gone = asyncio.Event()
    started = loop.time()
    pending = deque(payloads)
    parts: list[bytes] = []

    async def receive() -> Message:
        if pending:
            chunk = pending.popleft()
            return {"type": "http.request", "body": chunk, "more_body": bool(pending)}
        await gone.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        if gone.is_set() or response.completed:
            return
        if message["type"] == "http.response.start":
            response.status = int(message["status"])
            response.headers = {
                k.decode("latin-1").lower(): v.decode("latin-1")
                for k, v in message.get("headers") or ()
            }
        elif message["type"] == "http.response.body":
            chunk = message.get("body", b"")
            if chunk:
                if response.ttfb is None:
                    response.ttfb = loop.time() - started
                response.chunks += 1
                if keep_body:
                    parts.append(chunk)
            if not message.get("more_body", False):
                response.completed = True
            elif disconnect_after_chunks is not None and response.chunks >= disconnect_after_chunks:
                response.disconnected = True
                gone.set()

    def hang_up() -> None:
        if not response.completed:
            response.disconnected = True
            gone.set()

    timer = loop.call_later(disconnect_after_s, hang_up) if disconnect_after_s is not None else None
    try:
        await app(scope, receive, send)
    finally:
        if timer is not None:
            timer.cancel()
        response.duration = loop.time() - started
        response.body = b"".join(parts)
    return response


# --------------------------------------------------------------------------- load


def percentile(values: Iterable[float], q: float) -> float:
    """Nearest-rank percentile (``q`` in 0..100); ``nan`` for an empty input."""
    ordered = sorted(values)
    if not ordered:
        return math.nan
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return ordered[min(len(ordered), rank) - 1]


@dataclass(frozen=True, slots=True)
class TenantLoad:
    """Open-loop traffic from one tenant.

    Arrivals are a Poisson process at ``rps`` between ``start_s`` and ``stop_s``.  Each
    request is a chat completion with a prompt of about ``prompt_tokens`` tokens that asks
    for ``max_tokens`` and (against :class:`FakeModelBackend`) receives ``output_tokens``
    (default: ``max_tokens``), varied uniformly by +/- ``output_jitter`` so that requests
    do not all finish in lock-step.
    The tenant is identified by the ``x-tenant-id`` header unless ``headers`` says
    otherwise.
    """

    name: str
    rps: float
    prompt_tokens: int = 200
    max_tokens: int = 100
    output_tokens: int | None = None
    output_jitter: float = 0.5
    stream: bool = True
    include_usage: bool = True
    priority: str | None = None
    path: str = "/v1/chat/completions"
    headers: Mapping[str, str] = field(default_factory=dict)
    start_s: float = 0.0
    stop_s: float | None = None


@dataclass(frozen=True, slots=True)
class RequestRecord:
    tenant: str
    sent_at: float
    status: int
    #: ``ok``, ``shed`` (429/503 before the app ran), ``timeout`` (client gave up) or ``error``.
    outcome: str
    latency: float
    ttfb: float | None
    queue_s: float
    shed_reason: str | None
    output_tokens: int


@dataclass(frozen=True, slots=True)
class TenantStats:
    """Aggregates over one tenant's requests (or over all of them)."""

    name: str
    duration_s: float
    sent: int
    ok: int
    shed: int
    timeouts: int
    errors: int
    #: Completed successfully *and* within the SLO.
    good: int
    good_tokens: int
    latency_p50: float
    latency_p99: float
    ttfb_p50: float
    ttfb_p99: float
    queue_p99: float

    @property
    def shed_rate(self) -> float:
        return self.shed / self.sent if self.sent else 0.0

    @property
    def goodput_rps(self) -> float:
        return self.good / self.duration_s if self.duration_s else 0.0

    @property
    def goodput_tps(self) -> float:
        """Output tokens per second delivered within the SLO."""
        return self.good_tokens / self.duration_s if self.duration_s else 0.0

    @property
    def goodput_fraction(self) -> float:
        """Share of this tenant's requests that completed within the SLO."""
        return self.good / self.sent if self.sent else 0.0


@dataclass(slots=True)
class LoadReport:
    """Everything :func:`simulate_load` measured."""

    duration_s: float
    slo_s: float | None
    records: list[RequestRecord]
    tenants: list[str]

    def _stats(self, name: str, records: list[RequestRecord]) -> TenantStats:
        ok = [r for r in records if r.outcome == "ok"]
        good = [r for r in ok if self.slo_s is None or r.latency <= self.slo_s]
        return TenantStats(
            name=name,
            duration_s=self.duration_s,
            sent=len(records),
            ok=len(ok),
            shed=sum(1 for r in records if r.outcome == "shed"),
            timeouts=sum(1 for r in records if r.outcome == "timeout"),
            errors=sum(1 for r in records if r.outcome == "error"),
            good=len(good),
            good_tokens=sum(r.output_tokens for r in good),
            latency_p50=percentile((r.latency for r in ok), 50),
            latency_p99=percentile((r.latency for r in ok), 99),
            ttfb_p50=percentile((r.ttfb for r in ok if r.ttfb is not None), 50),
            ttfb_p99=percentile((r.ttfb for r in ok if r.ttfb is not None), 99),
            queue_p99=percentile((r.queue_s for r in ok), 99),
        )

    def tenant(self, name: str) -> TenantStats:
        return self._stats(name, [r for r in self.records if r.tenant == name])

    @property
    def overall(self) -> TenantStats:
        return self._stats("all", self.records)

    def format(self) -> str:
        """A plain-text table, one row per tenant plus a total."""
        header = (
            f"{'tenant':<12}{'sent':>7}{'ok':>7}{'shed':>7}{'t/out':>7}{'shed%':>7}"
            f"{'p50 s':>8}{'p99 s':>8}{'ttfb99':>8}{'good/s':>8}{'tok/s':>8}"
        )
        lines = [header, "-" * len(header)]
        for stats in [*(self.tenant(name) for name in self.tenants), self.overall]:
            lines.append(
                f"{stats.name:<12}{stats.sent:>7}{stats.ok:>7}{stats.shed:>7}{stats.timeouts:>7}"
                f"{100 * stats.shed_rate:>6.1f}%"
                f"{stats.latency_p50:>8.2f}{stats.latency_p99:>8.2f}{stats.ttfb_p99:>8.2f}"
                f"{stats.goodput_rps:>8.2f}{stats.goodput_tps:>8.1f}"
            )
        return "\n".join(lines)


async def simulate_load(
    app: ASGIApp,
    tenants: Iterable[TenantLoad],
    *,
    duration_s: float,
    seed: int = 0,
    slo_s: float | None = None,
    client_timeout_s: float | None = None,
    send_timeout_header: bool = False,
) -> LoadReport:
    """Drive ``app`` with open-loop traffic and report what happened.

    Open loop means arrivals do not slow down when the server does - which is what real
    overload looks like, and what a closed-loop benchmark hides.

    Args:
        duration_s: how long requests keep arriving.  The call returns once every request
            has finished, so the report covers all of them.
        seed: seed for the arrival processes; the same seed gives the same arrivals.
        slo_s: a completed request counts towards *goodput* only if it finished within
            this many seconds of being sent.
        client_timeout_s: clients hang up after this long (counted as ``timeout``).
        send_timeout_header: also announce that timeout in ``x-request-timeout`` so the
            middleware can use it as a queue deadline.

    Run it with :func:`run_virtual` for a fast, deterministic simulation, or under a
    normal event loop to exercise the app in real time.
    """
    loads = list(tenants)
    loop = asyncio.get_running_loop()
    origin = loop.time()
    records: list[RequestRecord] = []
    tasks: list[asyncio.Task[None]] = []

    async def one(load: TenantLoad, output_tokens: int) -> None:
        sent_at = loop.time() - origin
        body: dict[str, Any] = {
            "model": "fake-model",
            "messages": [{"role": "user", "content": "x" * (4 * load.prompt_tokens)}],
            "max_tokens": load.max_tokens,
            "stream": load.stream,
        }
        if load.stream and load.include_usage:
            body["stream_options"] = {"include_usage": True}
        body["x_fake_output_tokens"] = output_tokens
        headers = {"x-tenant-id": load.name, **load.headers}
        if load.priority is not None:
            headers.setdefault("x-priority", load.priority)
        if send_timeout_header and client_timeout_s is not None:
            headers.setdefault("x-request-timeout", str(client_timeout_s))
        observer = ResponseObserver()
        try:
            response = await call_asgi(
                app,
                "POST",
                load.path,
                json_body=body,
                headers=headers,
                disconnect_after_s=client_timeout_s,
            )
        except Exception:
            records.append(
                RequestRecord(
                    load.name,
                    sent_at,
                    500,
                    "error",
                    loop.time() - origin - sent_at,
                    None,
                    0.0,
                    None,
                    0,
                )
            )
            return
        shed_reason = response.headers.get("x-admitq-shed-reason")
        if response.disconnected or not response.completed:
            outcome = "timeout" if response.disconnected else "error"
        elif shed_reason is not None or response.status in (429, 503):
            outcome = "shed"
        elif 200 <= response.status < 300:
            outcome = "ok"
        else:
            outcome = "error"
        tokens = 0
        if outcome == "ok":
            observer.on_start(
                response.status,
                [(k.encode(), v.encode()) for k, v in response.headers.items()],
            )
            observer.on_body(response.body)
            tokens = observer.result().completion_tokens or 0
        records.append(
            RequestRecord(
                tenant=load.name,
                sent_at=sent_at,
                status=response.status,
                outcome=outcome,
                latency=response.duration,
                ttfb=response.ttfb,
                queue_s=float(response.headers.get("x-admitq-queue-ms", "0")) / 1000.0,
                shed_reason=shed_reason,
                output_tokens=tokens,
            )
        )

    async def arrivals(load: TenantLoad, rng: random.Random) -> None:
        if load.rps <= 0:
            return
        stop = duration_s if load.stop_s is None else min(duration_s, load.stop_s)
        at = load.start_s + rng.expovariate(load.rps)
        while at < stop:
            delay = origin + at - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            base = load.max_tokens if load.output_tokens is None else load.output_tokens
            spread = max(0.0, min(1.0, load.output_jitter))
            wanted = round(base * rng.uniform(1.0 - spread, 1.0 + spread))
            tasks.append(loop.create_task(one(load, max(1, min(load.max_tokens, wanted)))))
            at += rng.expovariate(load.rps)

    generators = [
        loop.create_task(arrivals(load, random.Random(f"{seed}:{index}:{load.name}")))
        for index, load in enumerate(loads)
    ]
    await asyncio.gather(*generators)
    if tasks:
        await asyncio.gather(*tasks)
    records.sort(key=lambda r: r.sent_at)
    return LoadReport(
        duration_s=duration_s,
        slo_s=slo_s,
        records=records,
        tenants=[load.name for load in loads],
    )

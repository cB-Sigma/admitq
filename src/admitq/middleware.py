"""Pure-ASGI admission middleware.

Works with any ASGI application (FastAPI, Starlette, Litestar, Quart, a bare callable) and
imports none of them::

    app.add_middleware(
        InferenceAdmission,
        max_queue_s=2.0,
        max_inflight_tokens=64_000,
        tenant_tpm={"default": 200_000},
    )

Per inference request it:

1. buffers the body (up to ``max_body_bytes``) and estimates the token cost;
2. asks the controller for admission - which may queue, or shed with 429/503;
3. replays the body to the application and watches the response go by, counting usage;
4. releases capacity exactly once - on completion, error, cancellation or disconnect - and
   corrects the tenant's budget with what the request really cost.

Requests that do not match an inference route are passed through untouched.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import Awaitable, Callable, Iterable, Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from .config import AdmissionConfig, PriorityClass
from .controller import AdmissionController, Ticket, _Request
from .errors import Shed
from .estimators import Cost
from .request import KeyFunction, RequestInfo, api_key_tenant, header_priority
from .routes import RouteMatcher
from .usage import ResponseObserver

if TYPE_CHECKING:
    from .adaptive import AdaptiveLimit
    from .backends.base import BudgetBackend
    from .estimators import CostEstimator
    from .metrics import Recorder

__all__ = ["DEFAULT_TIMEOUT_HEADERS", "InferenceAdmission"]

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
# The public boundary is typed loosely on purpose.  Frameworks disagree on the exact
# types (Starlette uses mutable mappings, Litestar typed dicts, ...); ``Any`` here is what
# lets ``InferenceAdmission`` type-check as middleware in all of them.
ASGIApp = Callable[[Any, Any, Any], Awaitable[None]]

#: Request headers that carry the client's own timeout, mapped to the factor that converts
#: the value to seconds.  ``x-stainless-read-timeout`` is sent by the official OpenAI Python
#: SDK (its read timeout, in seconds);
#: ``x-envoy-expected-rq-timeout-ms`` is set by Envoy-based proxies.
DEFAULT_TIMEOUT_HEADERS: Mapping[str, float] = {
    "x-request-timeout": 1.0,
    "x-stainless-read-timeout": 1.0,
    "x-envoy-expected-rq-timeout-ms": 0.001,
}

_STREAM_USAGE_ENDPOINTS = frozenset({"chat", "completions"})


def _consume(task: asyncio.Future[Any]) -> None:
    if not task.cancelled():
        task.exception()


class _State:
    __slots__ = ("body_done", "cancelled_downstream", "disconnected", "started", "status")

    def __init__(self) -> None:
        self.started = False
        self.status = 0
        self.body_done = False
        self.disconnected = False
        self.cancelled_downstream = False


class InferenceAdmission:
    """Admission control for inference routes.

    Args:
        app: the ASGI application to protect.
        controller: an existing :class:`AdmissionController` to use (for sharing one
            between several mounts, or to keep a handle on it).  If omitted, one is built
            from the controller options below.
        routes: which paths are inference requests; see :class:`admitq.RouteMatcher`.
            Defaults to the OpenAI chat, completions, embeddings and responses paths.
        tenant_key: ``RequestInfo -> str``.  Defaults to a hash of the API key.
        priority_key: ``RequestInfo -> str | None``.  Defaults to the ``x-priority`` header.
        max_body_bytes: largest request body that is buffered and parsed.  Larger bodies
            are not held in memory: they are streamed through and costed from their size.
        timeout_headers: request headers carrying the client's timeout (name -> factor to
            seconds).  Pass ``{}`` to ignore client timeouts.
        response_headers: add ``x-admitq-*`` and ``x-ratelimit-*`` headers to responses.
        metrics_path: if set, ``GET`` on this path returns Prometheus text.
        inject_stream_usage: for streaming chat/completions requests that did not ask for
            it, add ``stream_options.include_usage`` so the engine reports exact usage.
            The client then receives one extra final chunk with empty ``choices``.
        cancel_on_disconnect: cancel the downstream application when the client
            disconnects, for applications that do not watch for disconnects themselves.
        disconnect_grace_s: with ``cancel_on_disconnect``, how long the application gets
            to wind down by itself before it is cancelled.
        max_response_inspect_bytes: largest non-streaming response parsed in full to find
            ``usage`` (beyond that only the tail is searched).

    Every other keyword argument (``max_inflight_tokens``, ``max_queue_s``,
    ``tenant_tpm``, ...) is a field of :class:`admitq.AdmissionConfig` with the same
    meaning and default; they cannot be combined with ``controller=``.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        controller: AdmissionController | None = None,
        routes: RouteMatcher | Mapping[str, str] | Iterable[str] | None = None,
        tenant_key: KeyFunction | None = None,
        priority_key: KeyFunction | None = None,
        max_body_bytes: int = 4 * 1024 * 1024,
        timeout_headers: Mapping[str, float] | None = None,
        response_headers: bool = True,
        metrics_path: str | None = None,
        inject_stream_usage: bool = False,
        cancel_on_disconnect: bool = False,
        disconnect_grace_s: float = 0.0,
        max_response_inspect_bytes: int = 1024 * 1024,
        # Controller configuration - the fields of AdmissionConfig, spelled out so that
        # editors and type checkers see them through ``app.add_middleware(...)``.
        max_inflight_tokens: int | None = None,
        max_inflight_requests: int | None = None,
        max_queue_s: float = 2.0,
        max_queue_depth: int | None = 1024,
        early_shed: bool = True,
        oversize: Literal["clamp", "reject"] = "clamp",
        tenant_tpm: Mapping[str, float] | float | None = None,
        tenant_rpm: Mapping[str, float] | float | None = None,
        tenant_weights: Mapping[str, float] | None = None,
        burst_s: float = 60.0,
        max_tenant_share: float | None = None,
        priorities: Mapping[str, int] | Sequence[PriorityClass] | None = None,
        default_priority: str | None = None,
        overload_status: int = 503,
        retry_after_min: float = 1.0,
        retry_after_max: float = 60.0,
        estimator: CostEstimator | None = None,
        backend: BudgetBackend | None = None,
        on_backend_error: Literal["open", "closed"] = "open",
        adaptive: AdaptiveLimit | None = None,
        recorders: Sequence[Recorder] = (),
        max_tenant_labels: int = 100,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if max_body_bytes < 0:
            raise ValueError("max_body_bytes must be >= 0")
        options: dict[str, Any] = {
            "max_inflight_tokens": max_inflight_tokens,
            "max_inflight_requests": max_inflight_requests,
            "max_queue_s": max_queue_s,
            "max_queue_depth": max_queue_depth,
            "early_shed": early_shed,
            "oversize": oversize,
            "tenant_tpm": tenant_tpm,
            "tenant_rpm": tenant_rpm,
            "burst_s": burst_s,
            "max_tenant_share": max_tenant_share,
            "default_priority": default_priority,
            "overload_status": overload_status,
            "retry_after_min": retry_after_min,
            "retry_after_max": retry_after_max,
            "estimator": estimator,
            "backend": backend,
            "on_backend_error": on_backend_error,
            "adaptive": adaptive,
            "recorders": recorders,
            "max_tenant_labels": max_tenant_labels,
            "clock": clock,
        }
        if tenant_weights is not None:
            options["tenant_weights"] = tenant_weights
        if priorities is not None:
            options["priorities"] = priorities
        if controller is not None:
            defaults = {field.name: field.default for field in dataclasses.fields(AdmissionConfig)}
            changed = sorted(name for name, value in options.items() if value != defaults[name])
            if changed:
                raise TypeError(
                    "pass either a controller or controller options, not both "
                    f"(got controller= and {', '.join(changed)})"
                )
            self.controller = controller
        else:
            self.controller = AdmissionController(**options)
        self.app = app
        self.routes = routes if isinstance(routes, RouteMatcher) else RouteMatcher(routes)
        self.tenant_key: KeyFunction = tenant_key or api_key_tenant()
        self.priority_key: KeyFunction = priority_key or header_priority()
        self.max_body_bytes = max_body_bytes
        self.timeout_headers = {
            name.lower(): factor
            for name, factor in (
                DEFAULT_TIMEOUT_HEADERS if timeout_headers is None else timeout_headers
            ).items()
        }
        self.response_headers = response_headers
        self.metrics_path = metrics_path
        self.inject_stream_usage = inject_stream_usage
        self.cancel_on_disconnect = cancel_on_disconnect
        self.disconnect_grace_s = disconnect_grace_s
        self.max_response_inspect_bytes = max_response_inspect_bytes

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        method: str = scope.get("method", "GET")
        path: str = scope.get("path", "")
        if self.metrics_path is not None and path == self.metrics_path and method == "GET":
            await self._send_metrics(send)
            return
        endpoint = self.routes.match(method, path)
        if endpoint is None:
            await self.app(scope, receive, send)
            return
        await self._handle(scope, receive, send, method, path, endpoint)

    # ------------------------------------------------------------------ request path

    async def _handle(
        self, scope: Scope, receive: Receive, send: Send, method: str, path: str, endpoint: str
    ) -> None:
        controller = self.controller
        headers: dict[str, str] = {}
        for raw_name, raw_value in scope.get("headers") or ():
            headers[raw_name.decode("latin-1").lower()] = raw_value.decode("latin-1")

        # 1. Read the body, within the cap.
        content_length: int | None = None
        try:
            content_length = int(headers["content-length"])
        except (KeyError, ValueError):
            content_length = None
        chunks: list[bytes] = []
        size = 0
        complete = False
        if content_length is None or content_length <= self.max_body_bytes:
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return  # the client left before finishing the request
                if message["type"] != "http.request":
                    continue
                chunk = message.get("body", b"")
                if chunk:
                    chunks.append(chunk)
                    size += len(chunk)
                if not message.get("more_body", False):
                    complete = True
                    break
                if size > self.max_body_bytes:
                    break
        body = b"".join(chunks) if len(chunks) != 1 else chunks[0]
        del chunks

        # 2. Work out who is asking and what it will cost.
        parsed: dict[str, Any] | None = None
        if complete and body:
            try:
                decoded = json.loads(body)
            except (ValueError, RecursionError):
                decoded = None
            if isinstance(decoded, dict):
                parsed = decoded
        info = RequestInfo(
            method=method, path=path, endpoint=endpoint, headers=headers, body=parsed, scope=scope
        )
        tenant = self.tenant_key(info) or "default"
        priority = self.priority_key(info)
        cost: Cost
        if parsed is not None:
            cost = controller.estimator.estimate(parsed, endpoint)
        else:
            cost = controller.estimator.estimate_raw(
                size if complete or content_length is None else max(size, content_length),
                endpoint,
            )
        timeout: float | None = None
        for name, factor in self.timeout_headers.items():
            raw = headers.get(name)
            if raw is None:
                continue
            try:
                value = float(raw) * factor
            except ValueError:
                continue
            if value > 0 and (timeout is None or value < timeout):
                timeout = value

        if (
            self.inject_stream_usage
            and parsed is not None
            and endpoint in _STREAM_USAGE_ENDPOINTS
            and parsed.get("stream") is True
        ):
            stream_options = parsed.get("stream_options")
            if not (isinstance(stream_options, dict) and stream_options.get("include_usage")):
                merged = dict(stream_options) if isinstance(stream_options, dict) else {}
                merged["include_usage"] = True
                body = json.dumps({**parsed, "stream_options": merged}).encode()
                scope = dict(scope)
                scope["headers"] = [
                    (name, value)
                    for name, value in scope.get("headers") or ()
                    if name.lower() != b"content-length"
                ] + [(b"content-length", str(len(body)).encode())]

        # 3. Admission.  While queued, watch for the client going away.
        watcher: asyncio.Future[Message] | None = None
        try:
            try:
                request = await controller._submit(tenant, cost, priority, timeout)
                if complete and (request.future is not None or self.cancel_on_disconnect):
                    watcher = asyncio.ensure_future(receive())
                    watcher.add_done_callback(_consume)
                if request.future is not None and not await self._wait_admitted(request, watcher):
                    return  # disconnected while queued; nothing to answer
            except Shed as shed:
                await self._send_shed(send, shed, controller, tenant)
                return
            ticket = Ticket(controller, request)
            await self._run(scope, receive, send, ticket, body, complete, watcher)
        finally:
            if watcher is not None and not watcher.done():
                watcher.cancel()

    async def _wait_admitted(
        self, request: _Request, watcher: asyncio.Future[Message] | None
    ) -> bool:
        """Wait for a queued request.  Returns ``False`` if the client disconnected first;
        raises :class:`Shed` if the queue gave up on it."""
        future = request.future
        assert future is not None
        controller = self.controller
        try:
            if watcher is not None:
                pending: set[asyncio.Future[Any]] = {future, watcher}
                await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                if (
                    not future.done()
                    and watcher.exception() is None
                    and watcher.result()["type"] == "http.disconnect"
                ):
                    controller._abandon(request)
                    return False
            await future
        except asyncio.CancelledError:
            controller._abandon(request)
            raise
        return True

    async def _run(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        ticket: Ticket,
        body: bytes,
        complete: bool,
        watcher: asyncio.Future[Message] | None,
    ) -> None:
        state = _State()
        observer = ResponseObserver(
            chars_per_token=self.controller.estimator.chars_per_token,
            max_inspect_bytes=self.max_response_inspect_bytes,
        )
        replayed = False

        async def app_receive() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": not complete}
            if complete and watcher is not None:
                # Shielded: the application may cancel its own receive() call (Starlette
                # does when a streaming response ends); the shared watcher must survive.
                message = await asyncio.shield(watcher)
            else:
                message = await receive()
            if message["type"] == "http.disconnect":
                state.disconnected = True
            return message

        def finish(outcome: str) -> None:
            if ticket.released:
                return
            observed = observer.result()
            if observed.source != "estimate":
                ticket.report_usage(
                    prompt_tokens=observed.prompt_tokens,
                    completion_tokens=observed.completion_tokens,
                    source=observed.source,
                )
            ticket.release(outcome)

        extra_headers = self._response_headers(ticket) if self.response_headers else []

        async def app_send(message: Message) -> None:
            kind = message["type"]
            if kind == "http.response.start":
                state.started = True
                state.status = int(message.get("status", 200))
                observer.on_start(state.status, message.get("headers") or ())
                if extra_headers:
                    message = dict(message)
                    message["headers"] = [*(message.get("headers") or ()), *extra_headers]
                await send(message)
            elif kind == "http.response.body":
                chunk = message.get("body", b"")
                if chunk:
                    ticket.first_byte()
                    observer.on_body(chunk)
                if message.get("more_body", False):
                    await send(message)
                    return
                state.body_done = True
                try:
                    await send(message)
                finally:
                    if state.status >= 500:
                        finish("error")
                    elif state.status >= 400:
                        finish("client_error")
                    else:
                        finish("ok")
            else:
                await send(message)

        try:
            if self.cancel_on_disconnect and watcher is not None:
                await self._run_cancellable(scope, app_receive, app_send, watcher, state)
            else:
                await self.app(scope, app_receive, app_send)
        except asyncio.CancelledError:
            finish("cancelled")
            raise
        except BaseException:
            finish("client_disconnect" if state.disconnected else "error")
            raise
        else:
            # The application returned.  If it never completed the response, something cut
            # it short: usually the client disconnecting.
            if state.disconnected or state.cancelled_downstream:
                finish("client_disconnect")
            else:
                finish("error")

    async def _run_cancellable(
        self,
        scope: Scope,
        app_receive: Receive,
        app_send: Send,
        watcher: asyncio.Future[Message],
        state: _State,
    ) -> None:
        """Run the application in a task so it can be cancelled when the client leaves."""
        app_task: asyncio.Future[None] = asyncio.ensure_future(
            self.app(scope, app_receive, app_send)
        )
        try:
            both: set[asyncio.Future[Any]] = {app_task, watcher}
            await asyncio.wait(both, return_when=asyncio.FIRST_COMPLETED)
            if (
                not app_task.done()
                and watcher.done()
                and not watcher.cancelled()
                and watcher.exception() is None
                and watcher.result()["type"] == "http.disconnect"
            ):
                state.disconnected = True
                if self.disconnect_grace_s > 0:
                    await asyncio.wait({app_task}, timeout=self.disconnect_grace_s)
                if not app_task.done():
                    state.cancelled_downstream = True
                    app_task.cancel()
            await asyncio.wait({app_task})
        except asyncio.CancelledError:
            app_task.cancel()
            await asyncio.wait({app_task})
            raise
        if app_task.cancelled():
            if state.cancelled_downstream:
                return
            raise asyncio.CancelledError
        error = app_task.exception()
        if error is not None:
            raise error

    # ------------------------------------------------------------------ responses

    def _response_headers(self, ticket: Ticket) -> list[tuple[bytes, bytes]]:
        headers = [
            (b"x-admitq-queue-ms", str(round(ticket.queue_time * 1000)).encode()),
            (b"x-admitq-estimated-tokens", str(ticket.cost.total).encode()),
        ]
        headers.extend(_budget_headers(ticket.limits.tpm, ticket.remaining_tokens, b"tokens"))
        headers.extend(_budget_headers(ticket.limits.rpm, ticket.remaining_requests, b"requests"))
        return headers

    async def _send_shed(
        self, send: Send, shed: Shed, controller: AdmissionController, tenant: str
    ) -> None:
        payload = json.dumps(shed.body()).encode()
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(payload)).encode()),
        ]
        headers.extend((k.encode(), v.encode()) for k, v in shed.headers().items())
        if self.response_headers:
            limits = controller.config.limits_for(tenant)
            if shed.reason == "tenant_tokens":
                headers.extend(_budget_headers(limits.tpm, 0.0, b"tokens"))
            elif shed.reason == "tenant_requests":
                headers.extend(_budget_headers(limits.rpm, 0.0, b"requests"))
        await send({"type": "http.response.start", "status": shed.status, "headers": headers})
        await send({"type": "http.response.body", "body": payload})

    async def _send_metrics(self, send: Send) -> None:
        payload = self.controller.metrics.render().encode()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"text/plain; version=0.0.4; charset=utf-8"),
                    (b"content-length", str(len(payload)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})


def _budget_headers(
    limit: float | None, remaining: float | None, what: bytes
) -> list[tuple[bytes, bytes]]:
    if limit is None:
        return []
    headers = [(b"x-ratelimit-limit-" + what, str(int(limit)).encode())]
    if remaining is not None:
        headers.append((b"x-ratelimit-remaining-" + what, str(int(remaining)).encode()))
    return headers

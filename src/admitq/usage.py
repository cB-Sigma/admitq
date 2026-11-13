"""Observe a response as it passes through and work out what it actually cost.

Three sources, in order of trust:

``usage``
    A ``usage`` object in a non-streaming JSON body, or in a streamed chunk (OpenAI-style
    servers send one as the final chunk when the request sets
    ``stream_options.include_usage``; the Responses API sends it in ``response.completed``).
``counted``
    No usage reported: streamed text deltas are counted and converted to tokens.
``estimate``
    Nothing usable was seen; the pre-admission estimate stands.

The observer never modifies, delays or re-frames the bytes it is shown.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["ObservedUsage", "ResponseObserver", "extract_usage"]

UsageSource = Literal["usage", "counted", "estimate"]

_MAX_LINE_BYTES = 1 << 20
_TAIL_BYTES = 8192
_USAGE_RE = re.compile(rb'"usage"\s*:\s*(\{(?:[^{}]|\{[^{}]*\})*\})')
_TEXT_KEYS = ("content", "reasoning_content", "reasoning", "refusal")


@dataclass(frozen=True, slots=True)
class ObservedUsage:
    """What a response turned out to cost.  ``None`` means "not observed"."""

    prompt_tokens: int | None
    completion_tokens: int | None
    source: UsageSource


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value >= 0:
        return int(value)
    return None


def extract_usage(obj: Any) -> tuple[int | None, int | None] | None:
    """Pull ``(prompt_tokens, completion_tokens)`` out of a decoded response object.

    Understands OpenAI chat/completions/embeddings (``prompt_tokens`` /
    ``completion_tokens`` / ``total_tokens``), the Responses API (``input_tokens`` /
    ``output_tokens``, possibly nested under ``response``), Ollama's native API
    (``prompt_eval_count`` / ``eval_count``) and llama.cpp's native ``/completion``
    (``tokens_evaluated`` / ``tokens_predicted``).  Returns ``None`` if there is none.
    """
    if not isinstance(obj, Mapping):
        return None
    usage = obj.get("usage")
    if not isinstance(usage, Mapping):
        response = obj.get("response")
        if isinstance(response, Mapping):
            usage = response.get("usage")
    if isinstance(usage, Mapping):
        prompt = _int(usage.get("prompt_tokens"))
        if prompt is None:
            prompt = _int(usage.get("input_tokens"))
        completion = _int(usage.get("completion_tokens"))
        if completion is None:
            completion = _int(usage.get("output_tokens"))
        total = _int(usage.get("total_tokens"))
        if completion is None and total is not None and prompt is not None:
            completion = max(0, total - prompt)
        if prompt is not None or completion is not None:
            return prompt, completion
    for prompt_key, completion_key in (
        ("prompt_eval_count", "eval_count"),
        ("tokens_evaluated", "tokens_predicted"),
    ):
        prompt = _int(obj.get(prompt_key))
        completion = _int(obj.get(completion_key))
        if prompt is not None or completion is not None:
            return prompt, completion
    return None


def _delta_chars(obj: Mapping[str, Any]) -> int:
    """Number of generated characters carried by one streamed event."""
    chars = 0
    choices = obj.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            text = choice.get("text")
            if isinstance(text, str):
                chars += len(text)
            delta = choice.get("delta")
            if isinstance(delta, Mapping):
                for key in _TEXT_KEYS:
                    value = delta.get(key)
                    if isinstance(value, str):
                        chars += len(value)
                tool_calls = delta.get("tool_calls")
                if isinstance(tool_calls, list):
                    for call in tool_calls:
                        function = call.get("function") if isinstance(call, Mapping) else None
                        if isinstance(function, Mapping):
                            for key in ("name", "arguments"):
                                value = function.get(key)
                                if isinstance(value, str):
                                    chars += len(value)
        return chars
    event_type = obj.get("type")
    if isinstance(event_type, str):
        # Responses API: response.output_text.delta, response.reasoning_text.delta, ...
        if event_type.endswith(".delta"):
            delta = obj.get("delta")
            if isinstance(delta, str):
                chars += len(delta)
        return chars
    # Native streaming shapes: llama.cpp {"content": ...}, Ollama {"response": ...} or
    # {"message": {"content": ...}}.
    for key in ("content", "response"):
        value = obj.get(key)
        if isinstance(value, str):
            chars += len(value)
    message = obj.get("message")
    if isinstance(message, Mapping):
        value = message.get("content")
        if isinstance(value, str):
            chars += len(value)
    return chars


def _header(headers: Iterable[tuple[bytes, bytes]], name: bytes) -> bytes:
    for key, value in headers:
        if key.lower() == name:
            return value
    return b""


class ResponseObserver:
    """Incrementally inspects one HTTP response.

    Feed it :meth:`on_start` once and :meth:`on_body` for every body chunk, then read
    :meth:`result`.  Memory use is bounded by ``max_inspect_bytes`` for buffered JSON
    bodies and by a single SSE line for streams.
    """

    def __init__(self, *, chars_per_token: float = 4.0, max_inspect_bytes: int = 1 << 20) -> None:
        self._cpt = chars_per_token
        self._max = max_inspect_bytes
        self.status = 0
        self.started = False
        self.body_bytes = 0
        self._mode: Literal["sse", "ndjson", "json", "opaque"] = "opaque"
        self._line = bytearray()
        self._buffer: bytearray | None = None
        self._tail = bytearray()
        self._usage: tuple[int | None, int | None] | None = None
        self._events = 0
        self._chars = 0

    def on_start(self, status: int, headers: Iterable[tuple[bytes, bytes]]) -> None:
        self.status = status
        self.started = True
        headers = list(headers)
        content_type = _header(headers, b"content-type").split(b";", 1)[0].strip().lower()
        encoding = _header(headers, b"content-encoding").strip().lower()
        if encoding not in (b"", b"identity") or not 200 <= status < 300:
            self._mode = "opaque"
        elif content_type == b"text/event-stream":
            self._mode = "sse"
        elif content_type in (b"application/x-ndjson", b"application/jsonl"):
            self._mode = "ndjson"
        elif content_type == b"application/json" or content_type.endswith(b"+json"):
            self._mode = "json"
            self._buffer = bytearray()
        else:
            self._mode = "opaque"

    def on_body(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.body_bytes += len(chunk)
        if self._mode in ("sse", "ndjson"):
            self._feed_lines(chunk)
        elif self._mode == "json":
            if self._buffer is not None:
                if len(self._buffer) + len(chunk) <= self._max:
                    self._buffer += chunk
                else:
                    # Too large to parse in full: keep only the tail, where servers put usage.
                    self._tail = self._buffer[-_TAIL_BYTES:]
                    self._buffer = None
            if self._buffer is None:
                self._tail += chunk
                if len(self._tail) > 2 * _TAIL_BYTES:
                    del self._tail[:-_TAIL_BYTES]

    def result(self) -> ObservedUsage:
        """Best available usage for what has been observed so far."""
        if self._mode == "json" and self._usage is None:
            self._finish_json()
        if self._mode in ("sse", "ndjson") and self._line:
            # A final line without a trailing newline.
            self._handle_line(bytes(self._line))
            self._line.clear()
        if self._usage is not None:
            return ObservedUsage(self._usage[0], self._usage[1], "usage")
        if self._mode in ("sse", "ndjson") and (self._events or self._chars):
            counted = max(self._events, math.ceil(self._chars / self._cpt))
            return ObservedUsage(None, counted, "counted")
        return ObservedUsage(None, None, "estimate")

    # -- internals --------------------------------------------------------------------

    def _finish_json(self) -> None:
        if self._buffer is not None:
            try:
                self._usage = extract_usage(json.loads(self._buffer))
            except (ValueError, RecursionError):
                self._tail = self._buffer[-_TAIL_BYTES:]
            else:
                return
            finally:
                self._buffer = None
        matches = _USAGE_RE.findall(bytes(self._tail))
        if matches:
            try:
                self._usage = extract_usage({"usage": json.loads(matches[-1])})
            except (ValueError, RecursionError):
                self._usage = None

    def _feed_lines(self, chunk: bytes) -> None:
        if b"\n" not in chunk:
            if len(self._line) + len(chunk) > _MAX_LINE_BYTES:
                self._line.clear()  # pathological line: give up on it rather than grow
            else:
                self._line += chunk
            return
        data = bytes(self._line) + chunk if self._line else chunk
        self._line.clear()
        lines = data.split(b"\n")
        rest = lines.pop()
        if rest and len(rest) <= _MAX_LINE_BYTES:
            self._line += rest
        for line in lines:
            if line:
                self._handle_line(line)

    def _handle_line(self, line: bytes) -> None:
        line = line.strip()
        if self._mode == "sse":
            if not line.startswith(b"data:"):
                return
            line = line[5:].strip()
        if not line.startswith(b"{"):
            return  # "[DONE]", comments, keep-alives
        try:
            obj = json.loads(line)
        except (ValueError, RecursionError):
            return
        if not isinstance(obj, Mapping):
            return
        usage = extract_usage(obj)
        if usage is not None:
            self._usage = usage
        chars = _delta_chars(obj)
        if chars:
            self._events += 1
            self._chars += chars

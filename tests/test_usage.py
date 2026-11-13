from __future__ import annotations

import json

import pytest

from admitq import ResponseObserver, extract_usage

SSE = [(b"content-type", b"text/event-stream; charset=utf-8")]
JSON = [(b"content-type", b"application/json")]


def sse(obj) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def chunk(text: str) -> dict:
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


def test_extract_usage_shapes():
    assert extract_usage({"usage": {"prompt_tokens": 3, "completion_tokens": 4}}) == (3, 4)
    assert extract_usage({"usage": {"prompt_tokens": 3, "total_tokens": 10}}) == (3, 7)
    assert extract_usage({"usage": {"input_tokens": 5, "output_tokens": 6}}) == (5, 6)
    assert extract_usage({"response": {"usage": {"input_tokens": 1, "output_tokens": 2}}}) == (1, 2)
    assert extract_usage({"prompt_eval_count": 8, "eval_count": 9}) == (8, 9)  # Ollama native
    assert extract_usage({"tokens_evaluated": 8, "tokens_predicted": 9}) == (8, 9)  # llama.cpp
    assert extract_usage({"usage": None}) is None
    assert extract_usage({"usage": {"prompt_tokens": -1, "completion_tokens": True}}) is None
    assert extract_usage([1, 2]) is None


def test_non_streaming_json_usage():
    obs = ResponseObserver()
    obs.on_start(200, JSON)
    body = json.dumps(
        {
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 2},
        }
    ).encode()
    obs.on_body(body[:10])
    obs.on_body(body[10:])
    result = obs.result()
    assert (result.prompt_tokens, result.completion_tokens, result.source) == (11, 2, "usage")


def test_json_without_usage_keeps_estimate():
    obs = ResponseObserver()
    obs.on_start(200, JSON)
    obs.on_body(b'{"choices": []}')
    assert obs.result().source == "estimate"


def test_large_json_body_finds_usage_in_the_tail():
    obs = ResponseObserver(max_inspect_bytes=1024)
    obs.on_start(200, JSON)
    payload = json.dumps(
        {
            "data": [{"embedding": [0.125] * 4096}],
            "usage": {"prompt_tokens": 77, "total_tokens": 77, "details": {"cached": 0}},
        }
    ).encode()
    assert len(payload) > 20_000
    for start in range(0, len(payload), 1000):
        obs.on_body(payload[start : start + 1000])
    result = obs.result()
    assert (result.prompt_tokens, result.completion_tokens, result.source) == (77, 0, "usage")
    assert obs.body_bytes == len(payload)


def test_truncated_json_falls_back_to_tail_search():
    obs = ResponseObserver()
    obs.on_start(200, JSON)
    obs.on_body(b'{"usage": {"prompt_tokens": 5, "completion_tokens": 6}, "choices": [{"mess')
    result = obs.result()
    assert (result.prompt_tokens, result.completion_tokens) == (5, 6)


def test_stream_with_usage_chunk():
    obs = ResponseObserver()
    obs.on_start(200, SSE)
    for text in ("Hel", "lo", " world"):
        obs.on_body(sse(chunk(text)))
    obs.on_body(sse({"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 3}}))
    obs.on_body(b"data: [DONE]\n\n")
    result = obs.result()
    assert (result.prompt_tokens, result.completion_tokens, result.source) == (9, 3, "usage")


def test_stream_without_usage_counts_deltas():
    obs = ResponseObserver(chars_per_token=4.0)
    obs.on_start(200, SSE)
    obs.on_body(sse({"choices": [{"delta": {"role": "assistant", "content": ""}}]}))
    for _ in range(10):
        obs.on_body(sse(chunk("tok ")))
    obs.on_body(sse({"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": None}))
    obs.on_body(b"data: [DONE]\n\n")
    result = obs.result()
    assert (result.prompt_tokens, result.completion_tokens, result.source) == (None, 10, "counted")


def test_counted_stream_uses_text_length_when_chunks_are_merged():
    # An engine that flushes several tokens per event: the event count under-counts.
    obs = ResponseObserver(chars_per_token=4.0)
    obs.on_start(200, SSE)
    obs.on_body(sse(chunk("word " * 40)))  # 200 characters in one event
    assert obs.result().completion_tokens == 50


def test_counted_stream_uses_event_count_for_dense_text():
    # CJK-like output: one character per token, far fewer than 4 chars per token.
    obs = ResponseObserver(chars_per_token=4.0)
    obs.on_start(200, SSE)
    for _ in range(20):
        obs.on_body(sse(chunk("中")))
    assert obs.result().completion_tokens == 20


def test_sse_split_at_arbitrary_byte_boundaries():
    stream = b"".join(sse(chunk("tok ")) for _ in range(25))
    stream += sse({"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 25}})
    for size in (1, 2, 3, 7, 64, 1000):
        obs = ResponseObserver()
        obs.on_start(200, SSE)
        for start in range(0, len(stream), size):
            obs.on_body(stream[start : start + size])
        assert obs.result().completion_tokens == 25, size
        assert obs.result().source == "usage"


def test_sse_crlf_comments_and_garbage_lines():
    obs = ResponseObserver()
    obs.on_start(200, SSE)
    obs.on_body(b": keep-alive\r\n\r\nevent: message\r\ndata: {not json}\r\n\r\n")
    obs.on_body(b"data:" + json.dumps(chunk("abcd")).encode() + b"\r\n\r\n")
    obs.on_body(b"data: 42\r\n\r\ndata: [1,2]\r\n\r\n")
    assert obs.result().completion_tokens == 1


def test_interrupted_stream_reports_what_was_seen():
    obs = ResponseObserver()
    obs.on_start(200, SSE)
    obs.on_body(sse(chunk("tok ")) + sse(chunk("tok ")) + b'data: {"choices": [{"delta": {"con')
    result = obs.result()
    assert (result.completion_tokens, result.source) == (2, "counted")


def test_final_line_without_newline_is_parsed():
    obs = ResponseObserver()
    obs.on_start(200, SSE)
    obs.on_body(b"data: " + json.dumps(chunk("abcdefgh")).encode())
    assert obs.result().completion_tokens == 2


def test_continuous_usage_stats_last_one_wins():
    obs = ResponseObserver()
    obs.on_start(200, SSE)
    for n in range(1, 6):
        event = chunk("tok ")
        event["usage"] = {"prompt_tokens": 7, "completion_tokens": n, "total_tokens": 7 + n}
        obs.on_body(sse(event))
    assert obs.result().completion_tokens == 5


def test_completions_and_reasoning_and_tool_call_deltas():
    obs = ResponseObserver(chars_per_token=1.0)
    obs.on_start(200, SSE)
    obs.on_body(sse({"choices": [{"text": "abc"}]}))
    obs.on_body(sse({"choices": [{"delta": {"reasoning_content": "de"}}]}))
    obs.on_body(
        sse(
            {
                "choices": [
                    {"delta": {"tool_calls": [{"function": {"name": "f", "arguments": "{}"}}]}}
                ]
            }
        )
    )
    assert obs.result().completion_tokens == 3 + 2 + 3


def test_responses_api_stream():
    obs = ResponseObserver(chars_per_token=4.0)
    obs.on_start(200, SSE)
    obs.on_body(sse({"type": "response.output_text.delta", "delta": "abcdefgh"}))
    assert obs.result().completion_tokens == 2
    obs.on_body(
        sse(
            {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 3, "output_tokens": 9}},
            }
        )
    )
    result = obs.result()
    assert (result.prompt_tokens, result.completion_tokens, result.source) == (3, 9, "usage")


def test_ndjson_native_streams():
    obs = ResponseObserver(chars_per_token=4.0)
    obs.on_start(200, [(b"content-type", b"application/x-ndjson")])
    obs.on_body(b'{"message": {"content": "abcd"}, "done": false}\n')
    obs.on_body(b'{"response": "abcd", "done": false}\n')
    assert obs.result().completion_tokens == 2
    obs.on_body(b'{"done": true, "prompt_eval_count": 12, "eval_count": 2}\n')
    result = obs.result()
    assert (result.prompt_tokens, result.completion_tokens, result.source) == (12, 2, "usage")


@pytest.mark.parametrize(
    ("status", "headers"),
    [
        (500, JSON),
        (200, [(b"content-type", b"application/json"), (b"content-encoding", b"gzip")]),
        (200, [(b"content-type", b"text/html")]),
        (200, []),
    ],
)
def test_opaque_responses_are_left_alone(status, headers):
    obs = ResponseObserver()
    obs.on_start(status, headers)
    obs.on_body(b'{"usage": {"prompt_tokens": 1, "completion_tokens": 1}}')
    assert obs.result().source == "estimate"


def test_pathological_long_line_does_not_grow_memory():
    obs = ResponseObserver()
    obs.on_start(200, SSE)
    for _ in range(40):
        obs.on_body(b"x" * 65536)  # 2.5 MiB without a newline
    assert len(obs._line) <= 1 << 20
    obs.on_body(b"\n" + sse(chunk("abcd")))
    assert obs.result().completion_tokens == 1

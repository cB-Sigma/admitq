from __future__ import annotations

import pytest

from admitq import BodyEstimator, CharEstimator, Cost, CostEstimator, TiktokenEstimator


def test_cost_total_and_validation():
    assert Cost(10, 5).total == 15
    with pytest.raises(ValueError, match="non-negative"):
        Cost(-1, 0)


def test_chat_messages_and_max_tokens():
    est = CharEstimator()
    body = {
        "messages": [
            {"role": "system", "content": "a" * 40},
            {"role": "user", "content": "b" * 400},
        ],
        "max_tokens": 100,
    }
    cost = est.estimate(body, "chat")
    assert cost.prompt_tokens == 10 + 100 + 2 * 4  # text plus per-message overhead
    assert cost.output_tokens == 100
    assert isinstance(est, CostEstimator)


def test_max_completion_tokens_wins_and_n_multiplies():
    est = CharEstimator()
    body = {"messages": [], "max_completion_tokens": 50, "max_tokens": 9999, "n": 3}
    assert est.estimate(body, "chat").output_tokens == 150


def test_default_output_when_unspecified_and_output_factor():
    assert CharEstimator(default_output_tokens=256).estimate({}, "chat").output_tokens == 256
    est = CharEstimator(output_factor=0.25, max_output_tokens=1000)
    assert est.estimate({"max_tokens": 100_000}, "chat").output_tokens == 250
    assert est.estimate({"max_tokens": 400}, "chat").output_tokens == 100


def test_multimodal_parts_and_tools():
    est = CharEstimator(tokens_per_image=700, tokens_per_message=0)
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "x" * 8},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            },
            {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "f"}}]},
        ],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        "max_tokens": 0,
    }
    cost = est.estimate(body, "chat")
    assert cost.output_tokens == 0
    assert cost.prompt_tokens > 2 + 700
    without_tools = est.estimate({**body, "tools": []}, "chat")
    assert without_tools.prompt_tokens < cost.prompt_tokens


def test_completions_prompt_shapes():
    est = CharEstimator()
    assert est.estimate({"prompt": "abcd" * 10, "max_tokens": 5}, "completions") == Cost(10, 5)
    # A batch of prompts generates one completion each.
    batch = est.estimate({"prompt": ["abcd", "abcdefgh"], "max_tokens": 5}, "completions")
    assert batch == Cost(3, 10)
    # Pre-tokenized input is counted exactly.
    assert est.estimate({"prompt": [1, 2, 3, 4, 5], "max_tokens": 1}, "completions") == Cost(5, 1)
    nested = est.estimate({"prompt": [[1, 2, 3], [4, 5]], "max_tokens": 2}, "completions")
    assert nested == Cost(5, 4)
    assert est.estimate({"prompt": [], "max_tokens": 2}, "completions") == Cost(0, 2)
    assert est.estimate({"prompt": "x", "best_of": 4, "n": 2, "max_tokens": 10}, "completions") == (
        Cost(1, 40)
    )


def test_embeddings_have_no_output():
    est = CharEstimator()
    assert est.estimate({"input": "abcd" * 100}, "embeddings") == Cost(100, 0)
    assert est.estimate({"input": ["abcd", "abcd"], "max_tokens": 50}, "embeddings") == Cost(2, 0)
    assert est.estimate_raw(4000, "embeddings") == Cost(1000, 0)


def test_responses_api_shapes():
    est = CharEstimator(tokens_per_message=0)
    body = {
        "instructions": "i" * 40,
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "u" * 80}]},
            {"type": "function_call_output", "output": "o" * 40},
        ],
        "max_output_tokens": 64,
    }
    assert est.estimate(body, "responses") == Cost(10 + 20 + 10, 64)
    assert est.estimate({"input": "abcd" * 5}, "responses").prompt_tokens == 5


def test_native_engine_spellings():
    est = CharEstimator()
    assert est.estimate({"prompt": "abcd", "n_predict": 77}, "other").output_tokens == 77
    assert (
        est.estimate({"prompt": "abcd", "options": {"num_predict": 33}}, "other").output_tokens
        == 33
    )
    # n_predict = -1 means "unlimited" in llama.cpp: fall back to the default.
    assert est.estimate({"prompt": "abcd", "n_predict": -1}, "other").output_tokens == 512


def test_garbage_bodies_do_not_raise():
    est = CharEstimator()
    for body in (
        {"messages": "not a list"},
        {"messages": [1, None, "str", {"content": 5}]},
        {"prompt": {"a": 1}},
        {"max_tokens": "many", "n": True},
        {"input": [{"content": [1, 2, {"type": "x"}]}]},
        {"max_tokens": 10.0},
        {"max_tokens": -5},
    ):
        cost = est.estimate(body, "chat")
        assert cost.prompt_tokens >= 0
        assert cost.output_tokens >= 0
    assert est.estimate({"max_tokens": 10.0}, "chat").output_tokens == 10


def test_estimate_raw_and_validation():
    est = CharEstimator(chars_per_token=2.0, default_output_tokens=10)
    assert est.estimate_raw(101, "chat") == Cost(51, 10)
    assert est.estimate_raw(-5, "chat") == Cost(0, 10)
    with pytest.raises(ValueError, match="chars_per_token"):
        CharEstimator(chars_per_token=0)
    with pytest.raises(ValueError, match="output_factor"):
        CharEstimator(output_factor=0)
    with pytest.raises(ValueError, match="default_output_tokens"):
        CharEstimator(default_output_tokens=-1)


def test_custom_counter():
    est = BodyEstimator(lambda text: len(text.split()), tokens_per_message=0)
    body = {"messages": [{"role": "user", "content": "one two three"}], "max_tokens": 1}
    assert est.estimate(body, "chat") == Cost(3, 1)


def test_tiktoken_estimator_counts_real_tokens():
    tiktoken = pytest.importorskip("tiktoken")
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # the encoding file is downloaded on first use
        pytest.skip(f"tiktoken encoding unavailable: {exc}")
    est = TiktokenEstimator("cl100k_base", tokens_per_message=0)
    text = "The quick brown fox jumps over the lazy dog. <|endoftext|>"
    body = {"messages": [{"role": "user", "content": text}], "max_tokens": 7}
    assert est.estimate(body, "chat") == Cost(len(enc.encode(text, disallowed_special=())), 7)

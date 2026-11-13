# Engine notes

What each engine does under overload by itself, which signals it exposes, and how it
reports usage - the facts admitq's design leans on.  Everything below was read from the
engines' sources or documentation at the versions given, in November 2026.  Engines move
quickly; check your version.

admitq is an independent project and is not affiliated with any of them.

## Summary

| Engine (version read) | Bounded queue of its own? | Queue-depth signal | Usage in streams |
|---|---|---|---|
| vLLM (v0.34.0rc2) | Opt-in: `--max-num-queued-reqs`, `--max-num-queued-tokens` (503); unbounded by default | `vllm:num_requests_waiting`, `vllm:num_requests_running` | final chunk with `stream_options.include_usage`; every chunk with `continuous_usage_stats`; `--enable-force-include-usage` |
| SGLang (v0.5.23) | Optional: `--max-queued-requests` (503) | `sglang:num_queue_reqs`, `sglang:num_running_reqs` | `stream_options.include_usage` (+ `continuous_usage_stats`) |
| llama.cpp server (b12323) | No: deferred tasks wait in an unbounded deque | `llamacpp:requests_deferred`, `llamacpp:requests_processing` (needs `--metrics`); `GET /slots` | `stream_options.include_usage`; native API reports `tokens_predicted` / `tokens_evaluated` |
| TGI (v3.3.7) | Yes: `--max-concurrent-requests` (default 128), 429 beyond | `tgi_queue_size`, `tgi_batch_current_size` | - (not checked) |
| Ollama (v0.40.16) | Yes: `OLLAMA_MAX_QUEUE` (default 512), 503 beyond | - (none found) | OpenAI route: `stream_options.include_usage`; native: `prompt_eval_count` / `eval_count` |

Where an engine has its own bound, it is a global count (of requests; in vLLM's case
also of prompt tokens in prefill): it does not know tenants, has no priorities or
fairness, and bounds the size of the backlog rather than the time spent in it.  That is the gap admitq fills; the
engine's own bound remains a useful last line of defence behind it.

## vLLM

* **Queue.**  The scheduler's waiting queue is `create_request_queue(policy)`
  (`vllm/v1/core/sched/scheduler.py`); for FCFS that is `FCFSRequestQueue`, a subclass of
  `collections.deque` with no maximum length (`vllm/v1/core/sched/request_queue.py`), and
  requests are enqueued without a bound check.  RFC
  [#18826](https://github.com/vllm-project/vllm/issues/18826) ("Controlling the maximum
  length of the waiting queue", open since May 2025) describes the consequence and
  proposes rejecting with 503.
* **Its own valves (recent versions).**  `--max-num-queued-reqs` caps requests in flight
  (waiting or running) and `--max-num-queued-tokens` caps the prompt tokens currently in
  prefill; beyond either, new requests get HTTP 503.  Both default to `None` - no limit -
  and the option's own documentation calls the first "primarily a coarse capacity valve"
  that "bounds vLLM's otherwise unbounded request queue" (`vllm/config/scheduler.py`;
  added by PR #49445, committed 2026-08-31).  They are global counts enforced in the API
  server.  If a global valve is all you need and you run only vLLM, use them and skip
  admitq; if you need tenants, fairness, priorities, a time bound or budgets, put admitq
  in front and keep the valve as the backstop.
* **Signals.**  `vllm:num_requests_running` and `vllm:num_requests_waiting` gauges,
  `vllm:time_to_first_token_seconds` and `vllm:request_queue_time_seconds` histograms
  (`vllm/v1/metrics/loggers.py`).  Issue
  [#49538](https://github.com/vllm-project/vllm/issues/49538) reports that the gauges lag
  the true queue state by seconds when read through `/metrics`, which is why admitq's
  default adaptive signal is its own time-to-first-byte measurement rather than a scrape.
* **Usage.**  `stream_options.include_usage` adds a final usage chunk;
  `stream_options.continuous_usage_stats` puts usage in every chunk;
  `--enable-force-include-usage` forces both on (`should_include_usage` in
  `vllm/entrypoints/serve/utils/api_utils.py`).  admitq takes the last usage object it
  sees, so all three work, and a stream cut short under continuous stats is charged
  exactly.

## SGLang

* **Queue.**  `--max-queued-requests` ("The maximum number of queued requests";
  `python/sglang/srt/arg_groups/fields/schedule.py`).  When the waiting queue is full the
  scheduler aborts the incoming request with `503` and the message "The request queue is
  full." (`python/sglang/srt/managers/scheduler.py`).  Unset by default.
* **Signals.**  `sglang:num_queue_reqs`, `sglang:num_running_reqs`,
  `sglang:time_to_first_token_seconds`, `sglang:queue_time_seconds`
  (`python/sglang/srt/observability/metrics_collector.py`).
* **Usage.**  The OpenAI chat route honours `stream_options.include_usage` and
  `continuous_usage_stats` (`python/sglang/srt/entrypoints/openai/serving_chat.py`).

## llama.cpp server

* **Queue.**  Requests that find no free slot are deferred into
  `std::deque<server_task> queue_tasks_deferred` (`tools/server/server-queue.h`), which
  has no bound.  The number of slots is `--parallel`.
* **Signals.**  `GET /metrics` (only with `--metrics`) exposes
  `llamacpp:requests_processing` and `llamacpp:requests_deferred` among others.
  `GET /slots` (on by default, `--no-slots` to disable) returns per-slot state, and
  `?fail_on_no_slot=1` makes it answer 503 when no slot is free (`tools/server/README.md`).
* **Usage.**  OpenAI-compatible streams include usage when `include_usage` is set
  (`tools/server/server-task.cpp`); the native completion result carries
  `tokens_predicted` and `tokens_evaluated`, which admitq also understands.

**llama-cpp-python's server** (`python -m llama_cpp.server`, used in the real-engine
experiment) is a different program: one model instance, one generation at a time.  By
default (`interrupt_requests=True`) a new request interrupts the one in progress; with
`--interrupt_requests false` requests wait their turn without bound.

## Text Generation Inference

* **Queue.**  `--max-concurrent-requests` (default 128): "Having a low limit will refuse
  clients requests instead of having them wait for too long" (`launcher/src/main.rs`).
  Beyond it the router answers `429` (`InferError::Overloaded` in `router/src/infer/mod.rs`
  and `router/src/server.rs`).
* **Signals.**  `tgi_queue_size`, `tgi_batch_current_size`
  (`docs/source/reference/metrics.md`).

## Ollama

* **Queue.**  `OLLAMA_MAX_QUEUE` (default 512) bounds pending requests; beyond it the
  server answers `503` "server busy, please try again. maximum pending requests exceeded"
  (`envconfig/config.go`, `server/sched.go`, `server/routes.go`).  `OLLAMA_NUM_PARALLEL`
  (default 1) is the per-model concurrency.
* **Usage.**  The OpenAI-compatible route supports `stream_options.include_usage`
  (`openai/openai.go`); the native API reports `prompt_eval_count` and `eval_count`
  (`api/types.go`).

## Using an engine signal

```python
from admitq import AdaptiveLimit, AdmissionController
from admitq.signals import poll_signal, vllm_requests_waiting

controller = AdmissionController(
    max_inflight_requests=32,
    adaptive=AdaptiveLimit(
        signal="reported", target=2.0, min_limit=2, max_limit=512, dimension="requests"
    ),
)


async def fetch() -> str:
    return (await http.get("http://vllm:8000/metrics")).text


task = asyncio.create_task(poll_signal(controller, fetch, parse=vllm_requests_waiting))
```

Parsers provided: `vllm_requests_waiting`, `sglang_queue_requests`,
`llamacpp_requests_deferred`, `tgi_queue_size`, and the generic
`prometheus_value(text, name)`.  admitq ships no HTTP client; `fetch` is yours.

## What admitq cannot see

It sits in front of the engine and observes requests and responses, nothing else.

* It does not know how the engine batches, how full the KV cache is, or whether prefix
  caching made a prompt nearly free.  "In-flight tokens" is a proxy for engine load built
  from estimates.
* It cannot preempt or reorder work that has already been admitted.
* It measures time to first *byte* of the response it is shown.  Behind a proxy that
  buffers, that is not time to first token.
* If the engine is shared with clients that bypass your ASGI app, admitq does not see
  their load at all (the adaptive limit will still react to its effect on latency).

# Configuration reference

Everything is a keyword argument.  The same names work in three places:

```python
app.add_middleware(InferenceAdmission, max_queue_s=2.0, ...)       # middleware
controller = AdmissionController(max_queue_s=2.0, ...)             # programmatic
controller = AdmissionController(AdmissionConfig(max_queue_s=2.0)) # explicit config object
```

Invalid values raise `admitq.ConfigError` (a `ValueError`) at construction time.

## Capacity

| Option | Default | Meaning |
|---|---|---|
| `max_inflight_tokens` | `None` | Most estimated tokens (prompt + reserved output) in flight at once.  `None` = no token limit. |
| `max_inflight_requests` | `None` | Most requests in flight at once.  `None` = no request limit. |
| `oversize` | `"clamp"` | A request costing more than `max_inflight_tokens`: `"clamp"` charges it the whole limit so it runs alone; `"reject"` answers 400. |
| `max_tenant_share` | `None` | Hard cap on the fraction of in-flight tokens one tenant may hold (needs `max_inflight_tokens`).  A tenant with nothing in flight may always run one request. |
| `adaptive` | `None` | An `AdaptiveLimit` that moves one of the two limits; see below. |

With neither limit set, admitq only enforces tenant budgets.

**Choosing the numbers.**  Start from the engine: `max_inflight_requests` near what it
decodes concurrently (vLLM `--max-num-seqs`, llama.cpp `--parallel`, SGLang
`--max-running-requests`), and `max_inflight_tokens` near the context you are willing to
have resident at once.  Then watch `admitq_time_to_first_byte_seconds` (rises when the
limit is too high and work queues inside the engine) and `admitq_estimate_ratio` (far
below 1 means reservations are much larger than real usage, so the token limit is tighter
than it looks).  Or let `AdaptiveLimit` find it.

## Queue

| Option | Default | Meaning |
|---|---|---|
| `max_queue_s` | `2.0` | Longest a request may wait for capacity.  `0` = never queue. |
| `max_queue_depth` | `1024` | Most requests waiting at once; `None` = bounded by time only. |
| `early_shed` | `True` | Refuse on arrival when the predicted wait already exceeds the request's deadline. |
| `overload_status` | `503` | Status for overload sheds; `429` if your clients only back off on that. |
| `retry_after_min` / `retry_after_max` | `1.0` / `60.0` | Clamp for `Retry-After` on overload sheds (seconds). |

**Keep the queue bound well inside your clients' patience.**  Under sustained overload
a queue runs full, so a typical admitted request has already waited most of
`max_queue_s` before it starts.  If that plus the generation time exceeds the client's
timeout, the engine does the work and the client is no longer there to receive it.  Two
seconds for interactive traffic and a fraction of the client timeout for batch traffic
are sane starting points; clients that send a timeout header are handled automatically.

## Tenants

| Option | Default | Meaning |
|---|---|---|
| `tenant_tpm` | `None` | Tokens per minute.  `{"default": 200_000, "acme": 1_000_000}`; `"default"` applies to every tenant not listed, each with its own bucket.  A bare number means `{"default": n}`. |
| `tenant_rpm` | `None` | Requests per minute, same shape. |
| `tenant_weights` | `{}` | Fair-queuing weights (default 1; `"default"` key supported).  A tenant with weight 3 gets three times the service of a weight-1 tenant when both are waiting. |
| `burst_s` | `60.0` | Bucket size in seconds of refill.  60 = a full minute's budget may be spent at once; 10 = at most ten seconds' worth. |
| `backend` | in-memory | Where budgets live; see [backends.md](backends.md). |
| `on_backend_error` | `"open"` | If the backend fails: `"open"` skips the budget check, `"closed"` sheds with 503. |

## Priorities

| Option | Default | Meaning |
|---|---|---|
| `priorities` | `interactive`=0, `batch`=1 | `{name: level}` or a list of `PriorityClass`.  Lower level is served first. |
| `default_priority` | lowest level | Class for requests naming none, or an unknown one. |

```python
PriorityClass(
    name="batch",
    level=1,
    max_queue_s=30.0,  # this class may wait longer than the global max_queue_s
    aging_s=None,  # None: half of its queue deadline; math.inf: strict priority
)
```

`aging_s` is the starvation protection: a waiting request is treated as one level more
urgent for every `aging_s` seconds it has waited.

## Estimation

| Option | Default | Meaning |
|---|---|---|
| `estimator` | `CharEstimator()` | Anything implementing `CostEstimator`. |

```python
CharEstimator(
    chars_per_token=4.0,  # ~4 for English; lower for code, far lower for CJK
    default_output_tokens=512,  # assumed when the request sets no max_tokens
    output_factor=1.0,  # fraction of the requested max output to reserve
    max_output_tokens=None,  # cap on reserved output, e.g. the model's limit
    tokens_per_message=4,  # chat-template overhead per message
    tokens_per_image=512,  # flat cost of a non-text content part
)
TiktokenEstimator("cl100k_base", **same_keywords)  # pip install admitq[tiktoken]
TokenizersEstimator("tokenizer.json", **same_keywords)  # pip install admitq[tokenizers]
BodyEstimator(count_text=my_counter, **same_keywords)  # bring your own str -> int
```

`output_factor` deserves a thought.  At 1.0 every request reserves its full `max_tokens`
- safe, but clients that always send `max_tokens=4096` and use 200 will make the token
limit look twenty times tighter than it is.  The reservation is corrected when the
response ends, but capacity is held at the estimate while it runs.

## Middleware-only options

| Option | Default | Meaning |
|---|---|---|
| `controller` | `None` | Use an existing `AdmissionController` (cannot be combined with controller options). |
| `routes` | OpenAI paths | `{path: kind}`, a list of paths, or a `RouteMatcher`.  Paths match by suffix (so mount prefixes work); keys starting with `^` are regexes.  Kinds: `chat`, `completions`, `embeddings`, `responses`, `other`. |
| `tenant_key` | `api_key_tenant()` | `RequestInfo -> str`. |
| `priority_key` | `header_priority("x-priority")` | `RequestInfo -> str | None`. |
| `max_body_bytes` | 4 MiB | Largest body buffered and parsed; larger ones stream through. |
| `timeout_headers` | see below | Request headers carrying the client's timeout, `{name: factor to seconds}`; `{}` to ignore. |
| `response_headers` | `True` | Add `x-admitq-*` and `x-ratelimit-*` headers. |
| `metrics_path` | `None` | Serve Prometheus text on `GET` of this path. |
| `inject_stream_usage` | `False` | Add `stream_options.include_usage` to streaming chat/completions requests that lack it.  The client then sees one extra final chunk with empty `choices`. |
| `cancel_on_disconnect` | `False` | Cancel the app when the client disconnects. |
| `disconnect_grace_s` | `0.0` | Grace period before that cancellation. |
| `max_response_inspect_bytes` | 1 MiB | Largest JSON response parsed in full for `usage`. |

Default `timeout_headers`: `x-request-timeout` (s), `x-stainless-read-timeout` (s),
`x-envoy-expected-rq-timeout-ms` (ms).

Default routes: `POST` to a path ending in `/v1/chat/completions`, `/v1/completions`,
`/v1/embeddings` or `/v1/responses`.  For native engine APIs:

```python
routes = {
    "/v1/chat/completions": "chat",
    "/completion": "other",  # llama.cpp native
    "/api/generate": "other",  # Ollama native
    "/api/chat": "chat",
    "/generate_stream": "other",  # TGI native
}
```

### Tenant and priority keys

```python
from admitq import (
    api_key_tenant,
    header_tenant,
    body_field_tenant,
    jwt_claim_tenant,
    header_priority,
)

tenant_key = api_key_tenant()  # sha256 prefix of the bearer token / x-api-key (default)
tenant_key = header_tenant("x-tenant-id")  # a header your edge sets
tenant_key = body_field_tenant("user")  # OpenAI's `user` field
tenant_key = jwt_claim_tenant("org_id")  # a JWT claim - decoded, NOT verified
tenant_key = lambda info: info.scope["state"]["org"]  # whatever your auth layer stored
```

A key function receives a `RequestInfo` (`method`, `path`, `endpoint`, lower-cased
`headers`, parsed `body` or `None`, the ASGI `scope`).  Tenant identity is exactly as
trustworthy as the function you choose; see [SECURITY.md](../SECURITY.md).

## The adaptive limit

```python
AdaptiveLimit(
    target=0.5,  # seconds of time-to-first-byte (or the unit of your signal)
    min_limit=1_000,
    max_limit=200_000,
    initial=None,  # default: the configured static limit
    dimension=None,  # "tokens" | "requests"; default: tokens if a token limit is set
    signal="ttfb",  # or "reported": only values passed to controller.report_signal()
    interval_s=10.0,  # one decision per window
    quantile=0.9,
    increase=None,  # additive step; default: one typical request
    decrease=0.85,  # multiplicative back-off
    min_samples=5,
    utilization=0.8,  # only grow if the limit was at least this much in use
)
```

## Observability

| Option | Default | Meaning |
|---|---|---|
| `recorders` | `()` | Extra metric sinks (`PrometheusRecorder`, `OpenTelemetryRecorder`, your own). |
| `max_tenant_labels` | `100` | Distinct tenants that get their own metric series; the rest are `_other`. |
| `clock` | loop time | `() -> float` monotonic seconds, for tests. |

## The programmatic API

```python
controller = AdmissionController(max_inflight_tokens=64_000, max_queue_s=2.0)

cost = controller.estimate(body, "chat")  # -> Cost(prompt_tokens, output_tokens)

async with controller.admit(tenant="acme", cost=cost, priority="batch", timeout=30) as ticket:
    ...  # raises admitq.Shed if refused
    ticket.first_byte()  # optional: time-to-first-byte
    ticket.report_usage(prompt_tokens=..., completion_tokens=...)

ticket = await controller.acquire(tenant="acme", cost=1200)  # the same, by hand
try:
    ...
finally:
    ticket.release(
        "ok"
    )  # idempotent; "error" / "cancelled" charge nothing unless usage was reported

controller.snapshot()  # in-flight / queued / limits / mean hold time
controller.report_signal(x)  # feed an engine signal to AdaptiveLimit(signal="reported")
await controller.aclose()  # shed what is queued, flush budget corrections, close the backend
```

`Shed` carries `.reason`, `.status`, `.retry_after`, `.tenant`, `.body()` (OpenAI-style
error object) and `.headers()`.

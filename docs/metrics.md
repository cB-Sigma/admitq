# Metrics

Every controller has a built-in registry.  Three ways to get at it:

```python
# 1. Let the middleware serve it
app.add_middleware(InferenceAdmission, metrics_path="/admitq/metrics", ...)

# 2. Mount it yourself
from admitq import AdmissionController, metrics_app
controller = AdmissionController(...)
app.add_middleware(InferenceAdmission, controller=controller)
app.mount("/metrics/admitq", metrics_app(controller))

# 3. Read it in code or tests
controller.metrics.value("admitq_inflight_tokens")
controller.metrics.total("admitq_shed_total", reason="queue_timeout")
print(controller.metrics.render())      # Prometheus text format 0.0.4
```

The built-in exposition has no dependencies.  It is **per process**: with several
workers, scrape each one (or use `prometheus_client`'s multiprocess mode through the
recorder below).

## Reference

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `admitq_admitted_total` | counter | `tenant`, `priority` | Requests admitted to the app |
| `admitq_queued_total` | counter | `tenant`, `priority` | Requests that had to wait first (admitted or shed later) |
| `admitq_shed_total` | counter | `tenant`, `priority`, `reason` | Requests refused |
| `admitq_completed_total` | counter | `tenant`, `outcome` | Admitted requests that finished |
| `admitq_queue_time_seconds` | histogram | `priority` | Time admitted requests waited for capacity |
| `admitq_time_to_first_byte_seconds` | histogram | - | Admission to first response body byte |
| `admitq_estimate_ratio` | histogram | `source` | Actual tokens / estimated tokens, where usage was observed |
| `admitq_estimated_tokens_total` | counter | `tenant` | Tokens reserved at admission |
| `admitq_actual_tokens_total` | counter | `tenant`, `source` | Tokens charged after correction |
| `admitq_inflight_tokens` | gauge | - | Estimated tokens in flight |
| `admitq_inflight_requests` | gauge | - | Requests in flight |
| `admitq_inflight_token_limit` | gauge | - | Current token limit (moves with the adaptive limit; 0 = none) |
| `admitq_inflight_request_limit` | gauge | - | Current request limit (0 = none) |
| `admitq_queue_depth` | gauge | - | Requests waiting |
| `admitq_queued_tokens` | gauge | - | Estimated tokens waiting |
| `admitq_backend_errors_total` | counter | `operation` | Budget backend calls that failed or timed out |

Label values:

* `reason`: `tenant_tokens`, `tenant_requests` (429) · `queue_timeout`, `predicted_wait`,
  `queue_full`, `backend_unavailable` (503 by default) · `too_large` (400).
* `outcome`: `ok`, `client_error` (4xx from the app), `error` (5xx or exception),
  `client_disconnect`, `cancelled`, `abandoned_in_queue` (left the queue before being
  admitted; not counted in `admitted_total`).
* `source`: `usage`, `counted`, `estimate` (see [how-it-works.md](how-it-works.md#4-observing-the-response)).

Invariant you can alert on: `sum(admitted_total) = sum(completed_total{outcome!="abandoned_in_queue"}) + inflight_requests`.

## Tenant label cardinality

Tenant names become label values, and tenants can be unbounded (the default key is a hash
of the API key).  The first `max_tenant_labels` (100) distinct tenants get their own
series; everyone after that is reported as `tenant="_other"`.  This only affects labels -
budgets and fairness are always per real tenant.

## Things worth graphing

* `rate(admitq_shed_total[5m])` by `reason`: `tenant_*` means clients exceeding budgets;
  `predicted_wait`/`queue_timeout` means the engine is the bottleneck.
* `histogram_quantile(0.5, admitq_estimate_ratio_bucket)`: far below 1 means you reserve
  much more than requests use (lower `output_factor` or raise the in-flight token limit);
  above 1 means the estimator under-counts (check `chars_per_token`).
* `admitq_inflight_tokens / admitq_inflight_token_limit`: pinned at 1 with an empty queue
  is healthy saturation; pinned at 1 with constant shedding is overload.
* `admitq_time_to_first_byte_seconds` against `admitq_queue_time_seconds`: if time to
  first byte grows while queue time stays flat, the in-flight limit is too high and the
  queue has moved into the engine.

## prometheus_client

```python
from admitq.contrib.prometheus import PrometheusRecorder

app.add_middleware(InferenceAdmission, recorders=[PrometheusRecorder()], ...)
```

Same metric names and labels, registered in the default (or a given) `CollectorRegistry`,
exported by whatever exposition you already use.  Create one recorder per registry.

## OpenTelemetry

```python
from admitq.contrib.opentelemetry import OpenTelemetryRecorder

app.add_middleware(InferenceAdmission, recorders=[OpenTelemetryRecorder()], ...)
```

Instruments are created on the global meter provider (or the meter you pass) as
`admitq.admitted`, `admitq.shed`, `admitq.queue_time` (unit `s`), `admitq.actual_tokens`
(unit `{token}`) and so on; gauges are observable gauges reporting the latest value.

## Response headers

With `response_headers=True` (default), admitted responses carry:

| Header | Meaning |
|---|---|
| `x-admitq-queue-ms` | time this request waited for capacity |
| `x-admitq-estimated-tokens` | the pre-admission estimate |
| `x-ratelimit-limit-tokens` / `-requests` | the tenant's per-minute budgets, if any |
| `x-ratelimit-remaining-tokens` / `-requests` | what is left after this request's reservation |

Shed responses carry `retry-after`, `x-admitq-shed-reason` and, for budget sheds, the
matching `x-ratelimit-*` pair with `remaining` at 0.

## Writing a recorder

```python
class Recorder(Protocol):
    def inc(self, name: str, labels: Mapping[str, str], value: float = 1.0) -> None: ...
    def observe(self, name: str, labels: Mapping[str, str], value: float) -> None: ...
    def set(self, name: str, labels: Mapping[str, str], value: float) -> None: ...
```

`admitq.METRICS` maps each name to its type, help text, label names and buckets.

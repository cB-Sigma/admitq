# State, backends and what is consistent across workers

admitq keeps two kinds of state, and they have different homes.

| State | Lives in | Shared across workers? |
|---|---|---|
| In-flight tokens / requests, the wait queue, fair-queuing clocks, drain-rate estimate, adaptive limit | the controller, in process memory | **never** |
| Tenant TPM / RPM budgets | the budget backend | in-memory: no · Redis: yes |
| Metrics | the built-in registry, in process memory | no (aggregate in Prometheus) |

## The in-memory backend (default)

`MemoryBackend` is a dict of token buckets.  It is exact and fast (no awaits on the
request path), and it is **per process**.

With `uvicorn --workers 4`, or four replicas behind a load balancer, each process has its
own buckets, so:

* a tenant with `tenant_tpm=200_000` can use up to **4 x 200 000** tokens per minute in
  total - more precisely, 200 000 per process it happens to be routed to;
* a tenant that is refused by one worker may be admitted by the next;
* `x-ratelimit-remaining-*` headers describe the bucket of the worker that answered.

If budgets are a soft guard against runaway clients, dividing the limit by the number of
workers is often good enough.  If they are a contract, use Redis.

## The Redis backend

```python
from admitq.backends.redis import RedisBackend

app.add_middleware(
    InferenceAdmission,
    backend=RedisBackend.from_url("redis://redis:6379/0"),
    tenant_tpm={"default": 200_000},
    ...
)
```

* One hash per tenant (`admitq:{tenant}`), holding both buckets and a timestamp.
* Refill, check and debit run in **one Lua script**, so concurrent workers cannot both
  spend the same budget.  The concurrency test fires 400 reservations from four
  connections at one bucket and checks that the number admitted is exactly what the
  bucket (plus refill during the test) can pay for.
* Time comes from the Redis server (`TIME`), so clock skew between workers is irrelevant.
* Keys expire after two burst windows of inactivity.
* The braces in the key are a Redis Cluster hash tag: each call touches one slot.
* Post-hoc corrections (`actual - estimate`) are applied in a background task after the
  response; `await controller.aclose()` flushes them on shutdown.

What Redis does **not** change: in-flight limits and the queue are still per process.
A wait queue shared between processes would need cross-process wake-ups and leases with
expiry for crashed workers; that is on the roadmap, not in this release.  In the
meantime give each worker its share:

```python
max_inflight_requests = ENGINE_MAX_SEQS // WORKERS
max_inflight_tokens = ENGINE_TOKEN_BUDGET // WORKERS
```

This is conservative when traffic is unevenly spread across workers (one worker may be
queueing while another has room).  A single-process proxy in front of the engine - which
is what an asyncio proxy usually is - has no such problem.

### When Redis is slow or down

Every call has a timeout (`timeout=0.25` seconds by default).  What a failure means is
your choice:

| `on_backend_error` | Behaviour |
|---|---|
| `"open"` (default) | The request skips the budget check and proceeds to the in-flight limits, which still protect the engine.  `admitq_backend_errors_total` counts it. |
| `"closed"` | The request is shed with `503` (`backend_unavailable`). |

### Tested against

The Redis tests run against a real `redis-server` (7.0.15 locally; the `redis:7` image in
CI).  The script uses only `TIME`, `HMGET`, `HSET` and `PEXPIRE`.  It also runs under
`fakeredis` with Lua support, which is convenient for your own unit tests, though only a
smoke check of that combination was done here.  Redis Cluster and Sentinel have not been
tested.

## Writing a backend

A backend is three coroutines:

```python
class BudgetBackend(Protocol):
    async def reserve(
        self, tenant: str, tokens: int, requests: int, limits: TenantLimits
    ) -> Reservation: ...
    async def adjust(
        self, tenant: str, tokens: int, requests: int, limits: TenantLimits
    ) -> None: ...
    async def aclose(self) -> None: ...
```

`reserve` must be atomic per tenant: check both buckets and debit both, or neither.
`adjust` applies a signed correction and must be safe to call after the fact.  The shared
arithmetic lives in `admitq.backends.base` (`bucket_reserve`, `bucket_adjust`) so a new
backend only has to provide storage and atomicity.  If your backend never blocks, also
implement `reserve_sync` / `adjust_sync` and the controller will skip the task hop.

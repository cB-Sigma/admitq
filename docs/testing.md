# Testing and tuning with `admitq.testing`

Admission settings are hard to reason about and expensive to learn about in production.
`admitq.testing` (standard library only) lets you put a configuration under simulated
load in an ordinary unit test.

## The pieces

| Helper | What it is |
|---|---|
| `FakeModelBackend` | An ASGI app that behaves like an inference server: `max_num_seqs` concurrent requests, `total_tps` tokens/s overall, `stream_tps` per request, `prefill_tps` for prompts, and an **unbounded** internal queue for the rest.  OpenAI routes, SSE, `usage`. |
| `run_virtual(coro)` | Runs a coroutine on `VirtualTimeLoop`: whenever the loop would sleep, the clock jumps to the next timer instead.  A ten-minute scenario takes about a second and is deterministic. |
| `simulate_load(app, tenants, duration_s=...)` | Open-loop Poisson traffic from several `TenantLoad`s; returns a `LoadReport`. |
| `call_asgi(app, method, path, ...)` | One request against any ASGI app, no HTTP library needed; can disconnect after N chunks or T seconds. |
| `FakeClock` | A clock you advance by hand, for driving `Scheduler` or `MemoryBackend` directly. |

## A tuning test

```python
from admitq import InferenceAdmission
from admitq.testing import FakeModelBackend, TenantLoad, run_virtual, simulate_load


def build():
    engine = FakeModelBackend(max_num_seqs=8, total_tps=240, stream_tps=30)
    return InferenceAdmission(
        engine,
        max_inflight_requests=8,
        max_inflight_tokens=6_000,
        max_tenant_share=0.5,
        max_queue_s=2.0,
        tenant_key=lambda info: info.headers.get("x-tenant-id", "anonymous"),
    )


def test_chat_survives_the_nightly_eval():
    report = run_virtual(
        simulate_load(
            build(),
            [
                TenantLoad("chat-ui", rps=1.0, prompt_tokens=200, max_tokens=60),
                TenantLoad("nightly-eval", rps=6.0, prompt_tokens=600, max_tokens=120),
            ],
            duration_s=300,
            slo_s=8.0,  # "good" = completed within 8 s
            client_timeout_s=30.0,  # clients hang up after 30 s
        )
    )
    chat = report.tenant("chat-ui")
    assert chat.goodput_fraction > 0.97, report.format()
    assert chat.latency_p99 < 8.0
```

`examples/tune_in_tests.py` is a runnable version.

## Reading a `LoadReport`

`report.tenant(name)` and `report.overall` return `TenantStats`:

| Field | Meaning |
|---|---|
| `sent`, `ok`, `shed`, `timeouts`, `errors` | request counts by outcome |
| `shed_rate` | `shed / sent` |
| `latency_p50`, `latency_p99` | end-to-end latency of **completed** requests |
| `ttfb_p50`, `ttfb_p99` | time to first byte of completed requests |
| `queue_p99` | time spent in admitq's queue (from `x-admitq-queue-ms`) |
| `good`, `goodput_rps`, `goodput_tps`, `goodput_fraction` | completed **within `slo_s`**: count, per second, output tokens per second, share of sent |

`report.format()` prints a table; `report.records` has one `RequestRecord` per request if
you want your own statistics.

Latency percentiles are over completed requests only, so a configuration that sheds
everything slow looks fast.  Always read them next to the shed rate and goodput.

## Open loop, on purpose

`simulate_load` generates arrivals from a clock, not from responses: clients do not slow
down because the server did.  That is what overload looks like in production (the users
keep coming), and it is what closed-loop benchmarks - N workers each waiting for a reply
before sending the next request - cannot show, because they throttle themselves.

## Modelling your engine

`FakeModelBackend` has four numbers.  Rough ways to get them:

| Parameter | From |
|---|---|
| `max_num_seqs` | the engine's concurrency setting |
| `total_tps` | output tokens/s you measure at saturation |
| `stream_tps` | output tokens/s of a single request on an idle engine |
| `prefill_tps` | prompt tokens/s (prompt length / time to first token, unloaded) |

It is a model.  It has no KV-cache pressure, no prefix caching, no preemption and no
dependence of speed on context length.  Use it to compare configurations and to catch
gross mistakes (a queue bound shorter than your longest request; a share cap that starves
the main tenant), then confirm on the real thing.

## Driving the scheduler directly

For exact, clock-controlled tests of ordering and deadlines, skip asyncio altogether:

```python
from admitq import Scheduler

s = Scheduler(limit_tokens=1000)
a, _ = s.offer(tenant="a", cost=800, now=0.0, deadline=2.0)
b, _ = s.offer(tenant="b", cost=800, now=0.1, deadline=2.1)  # queued
effects = s.release(a, now=1.0)
assert effects.admitted == [b] and b.queue_time == 0.9
```

## Real time

Everything above also runs on a normal event loop if you `await` it there instead of
using `run_virtual` - useful for exercising a real ASGI app with real I/O.  It then takes
as long as it says.

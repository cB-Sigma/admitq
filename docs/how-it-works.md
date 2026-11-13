# How it works

One inference request passes through five stages.  Each is small on its own; the value is
in how they fit together.

```
request ──► 1 estimate ──► 2 budget ──► 3 queue + capacity ──► app / engine ──► 4 observe ──► 5 settle
              cost          (tenant)      (global, fair)                         usage         release once,
                              │                 │                                              correct budget
                            429 +             503 +
                          Retry-After       Retry-After
```

Requests that do not match an inference route skip all of it: the middleware hands the
original `scope`, `receive` and `send` straight to your app.

## 1. Estimating the cost

Before a request can be admitted it needs a price.  The middleware buffers the JSON body
(up to `max_body_bytes`, 4 MiB by default), parses it once and asks the estimator for a
`Cost(prompt_tokens, output_tokens)`:

| Part | Where it comes from |
|---|---|
| prompt | `messages` (text parts, tool calls), `prompt`, `input`, `instructions`, `tools`, `response_format`; pre-tokenized inputs (`[1, 2, 3]`) are counted exactly |
| output | `max_completion_tokens`, else `max_tokens`, else `max_output_tokens`, else `default_output_tokens` (512); multiplied by `n` / `best_of` and by the number of prompts in a batch; scaled by `output_factor` |
| embeddings | prompt only - no output is reserved |

The default `CharEstimator` divides characters by four.  That is crude and the library
says so everywhere; what makes it workable is stage 5.  `TiktokenEstimator` and
`TokenizersEstimator` count with a real tokenizer if you install the extra, and anything
with `estimate()` / `estimate_raw()` / `chars_per_token` can be plugged in.

Three cases where the body is not parsed:

* **Too large** (`Content-Length` above the cap, or a chunked upload that grows past it):
  the body is *not* held in memory.  What was read is replayed, the rest streams straight
  through, and the cost is estimated from the byte count.
* **Not JSON, or not a JSON object**: costed from its size and passed on.  Your app
  produces the 400/422, not admitq.
* **Client disconnects mid-upload**: the request is dropped; the app is never called.

The body your app receives is byte-for-byte what the client sent (unless you opt into
`inject_stream_usage`, which adds `stream_options.include_usage` and fixes
`Content-Length`).

## 2. Tenant budgets

If the tenant has a token-per-minute or request-per-minute budget, the estimate is
reserved against it first.  Each budget is a token bucket: it refills continuously at
`limit / 60` per second and holds at most `limit * burst_s / 60` (one minute's worth by
default).

* Both buckets are checked and debited atomically - or neither is.
* A refusal is a `429` whose `Retry-After` is **exact**: the time until the bucket has
  refilled enough for this request.  (The tests check that waiting exactly that long is
  enough, and that waiting slightly less is not.)
* A request larger than the whole bucket is admitted when the bucket is full and drives
  it negative; the debt is repaid by the refill.  Oversized requests are slowed down, not
  locked out forever.
* If the request is later shed by the queue, or the client disconnects while queued, the
  reservation is refunded in full.

## 3. Capacity, the queue and who goes next

Capacity is two counters with optional limits: **in-flight tokens** (the sum of the
estimates of everything currently running) and **in-flight requests**.  A request that
fits is admitted at once.  One that does not fit waits - but only under three conditions.

**The queue is bounded in time.**  Every waiting request has a deadline:
`min(max_queue_s of its priority class, the client's own timeout)`.  The client's timeout
is read from `x-request-timeout`, `x-stainless-read-timeout` (sent by the OpenAI Python
SDK) or `x-envoy-expected-rq-timeout-ms`.  When the deadline passes the request is shed
with `503` (`queue_timeout`).  A request is never admitted after its deadline, even if
capacity appears in the same instant.

**The queue does not accept what it cannot serve.**  On arrival the scheduler predicts
the wait.  It knows how long admitted work holds capacity on average (`T`, measured) and
how much is in flight (`N`), so by Little's law the system drains about `N / T` tokens per
second; the work ahead of the newcomer divided by that rate is its expected wait.  If that
already exceeds the deadline the request is refused immediately (`predicted_wait`)
instead of being made to wait for a timeout it was always going to hit.  "Work ahead"
respects the scheduling order below, so a quiet tenant is not turned away because of a
noisy tenant's backlog.  Until the first request has completed there is no rate to
extrapolate from and nothing is shed early.

**The queue is bounded in length** (`max_queue_depth`, 1024 by default) as a memory
backstop.  When it is full, the newest request of the *lowest* priority present is the
one that goes, so a queue full of batch work cannot lock out interactive arrivals.

`Retry-After` on these 503s is `predicted wait - queue budget`, clamped to
`[retry_after_min, retry_after_max]`: the time after which the backlog would have drained
far enough for this request to fit inside one queue budget.  It is an estimate from the
measured drain rate, not a promise.

### Order

When capacity frees up, the next request is chosen by, in order:

1. **Priority level**, lower first.  Starvation protection is built in: a waiting request
   is treated as one level more urgent for every `aging_s` it has waited (by default half
   of its class's queue deadline), so batch work spends the second half of its queue life
   competing as interactive.
2. **Weighted fair queuing between tenants** (start-time fair queuing on token cost).
   Each tenant has a virtual time that advances by `cost / weight` when one of its
   requests is admitted; the tenant with the smallest virtual time goes next.  A tenant
   that sends a burst advances its own clock and falls behind everyone else's; a tenant
   that was idle re-enters at the current global time, with no credit for the past and no
   debt.  The property tests check the textbook bound: for any two tenants that stay
   backlogged, the difference in weight-normalised service never exceeds one maximum-size
   request each.
3. **Arrival order.**

The chosen request is admitted only if it fits.  Smaller requests behind it do **not**
jump ahead: that would let a stream of small requests starve a large one indefinitely.
The cost is some idle capacity while a large request waits for room.

`max_tenant_share` adds a hard cap on top: no tenant may hold more than that fraction of
the in-flight tokens (a tenant with nothing in flight may always run one request).  Fair
queuing decides who is *next*; it cannot take back capacity already handed to a tenant
whose requests run for a long time.  The share cap keeps room free for others.  It is
deliberately not work-conserving.

## 4. Observing the response

Once admitted, the request goes to your app with the buffered body replayed.  The
middleware wraps `send` and watches the response go by without modifying or delaying it:

| Response | What is read | `source` label |
|---|---|---|
| JSON body | `usage` (`prompt_tokens`/`completion_tokens`, or `input_tokens`/`output_tokens`) | `usage` |
| SSE stream with a usage chunk | the last `usage` object seen (`stream_options.include_usage`, vLLM's `continuous_usage_stats`, Responses API `response.completed`) | `usage` |
| SSE stream without usage | text deltas are counted: `max(events with text, characters / chars_per_token)` | `counted` |
| anything else (compressed, non-2xx, unknown type) | nothing | `estimate` |

For JSON bodies up to `max_response_inspect_bytes` the whole body is parsed; beyond that
only the last 8 KiB are searched for a `usage` object (embedding responses are megabytes
of floats with the usage at the end).  Native streaming formats of Ollama
(`prompt_eval_count`/`eval_count`) and llama.cpp (`tokens_evaluated`/`tokens_predicted`)
are understood as well.

## 5. Settling

Capacity is released **exactly once**, at the first of:

* the final response body chunk has been sent;
* the app raised;
* the request task was cancelled;
* the client disconnected (the app returned or raised after `http.disconnect`).

Release is idempotent, so the several code paths that can reach it cannot double-free.
At release:

* the tenant's budget is corrected by `actual - estimate` (a refund if the estimate was
  high, an extra charge - bounded at one bucket of debt - if it was low);
* the tenant's fair-queuing clock is corrected by the same difference, so fairness
  follows real usage;
* the hold time feeds the drain-rate estimate used for early shedding;
* the next waiting requests are admitted.

What is charged when things go wrong:

| Outcome | Tokens charged |
|---|---|
| completed, usage observed | observed usage |
| completed, nothing observed | the estimate |
| stream cut short (disconnect, cancel, error) | prompt estimate + output counted so far |
| failed before any usage (4xx/5xx, exception, disconnect before first byte) | 0 (the request still counts against RPM) |

### Client disconnects

* **While queued**: the middleware is waiting on both the queue and `receive()`.  On
  `http.disconnect` the request is withdrawn, its reservation refunded, and the app is
  never called.
* **While streaming**: ASGI servers deliver `http.disconnect` through `receive()`;
  frameworks such as Starlette react by stopping the response generator.  The middleware
  releases when the app returns.  If your app does *not* watch for disconnects (a raw ASGI
  app, or an endpoint blocked on an upstream call that has not produced a byte yet), set
  `cancel_on_disconnect=True`: the middleware then runs the app in a task and cancels it
  when the client goes away, after an optional `disconnect_grace_s`.

One gap: a request whose body was too large to buffer cannot be watched for a disconnect
while it is *queued*, because the rest of its body is still unread on the same channel.
It leaves the queue by being admitted or by reaching its deadline.

A response is never shed after it has started: every shed decision is taken before your
app is called.

## The adaptive limit (optional)

A static in-flight limit is a guess about the engine.  `AdaptiveLimit` replaces the guess
with a feedback loop (AIMD):

* once per `interval_s`, take the chosen quantile (p90) of a congestion signal over the
  window;
* above `target`: multiply the limit by `decrease` (0.85);
* at or below `target` **and** the limit was actually in use: add one typical request to
  it (1 for a request limit; the running mean request cost for a token limit);
* otherwise leave it alone - an idle system teaches nothing.

The signal is either the time to first byte admitq measures itself (works with every
engine, needs nothing from it) or a number you report, such as the engine's queue-depth
gauge (see [engines.md](engines.md)).

In simulation (an engine that fits 8 requests) the limit settles into the band 7-9 from
starting points of 1, 4, 32 and 64.  A static limit of 64 on the same engine leaves
dozens of requests queued inside it.  Caveats, stated plainly: the signal lags by about
one prefill; `interval_s` must be several times the typical time to first byte; a target
below the engine's unloaded latency drives the limit to `min_limit`; and this has been
validated in simulation only, not against a GPU engine.

## Concurrency model

The scheduler is a synchronous state machine; the controller drives it from one event
loop.  Nothing in admitq blocks, spawns threads or takes locks.  One controller belongs
to one loop (and therefore one process); see [backends.md](backends.md) for what that
means with several workers.

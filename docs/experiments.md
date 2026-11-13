# Experiments

Two experiments, one simulated and one on a real engine.  Both are open-loop (arrivals do
not slow down when the server does) and both scripts are in [`benchmarks/`](../benchmarks);
raw outputs are in [`benchmarks/results/`](../benchmarks/results).

Read the numbers for what they are: evidence that the mechanisms do what they claim under
one workload each.  They are not a performance claim about your deployment.

Definitions used below:

* **completed** - the client received a full response;
* **shed** - refused by admitq with 429/503 before reaching the engine;
* **timed out** - the client gave up (after 30 s) and disconnected;
* **good** - completed within the SLO (10 s from send);
* latency percentiles are over completed requests only - always read them next to the
  shed and time-out counts.

## 1. Simulated overload with a noisy tenant

**SIMULATED.**  The engine is `admitq.testing.FakeModelBackend` and time is virtual.
Deterministic: the same command reproduces the same table.

```bash
python benchmarks/simulated_overload.py --seeds 5 --markdown
```

**Engine model:** 16 concurrent sequences, 800 output tokens/s in total, at most 40
tokens/s per request, prefill at 4000 tokens/s, unbounded internal queue.

**Traffic (300 s, Poisson):**

| Tenant | Rate | Prompt | Output | Share of engine it asks for |
|---|---|---|---|---|
| alice | 1.5 req/s | ~150 tokens | ~80 tokens | 15% |
| bob | 1.5 req/s | ~150 tokens | ~80 tokens | 15% |
| noisy | 5 req/s | ~1500 tokens | ~300 tokens | 190% |

**Arms:**

| Arm | Configuration |
|---|---|
| none | requests go straight to the engine |
| rpm-200 | request-count limit, 200/min per tenant (emulated: admitq with only `tenant_rpm=200`) |
| rpm-60 | the same limiter tightened to 60/min per tenant |
| admitq | `max_inflight_requests=16, max_inflight_tokens=32_000, max_queue_s=2.0, tenant_tpm=200_000` |
| admitq+share | the same plus `max_tenant_share=0.5` |

200 requests/minute is what one might pick for this engine if every request looked like
alice's.  The noisy tenant (300/minute) exceeds it, so the limiter does act.

### Results (seed 0)

| arm | tenant | sent | completed | shed | timed out | p50 s | p99 s | TTFB p99 s | good req/s | good tok/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | alice | 435 | 238 | 0 (0%) | 197 | 29.3 | 30.0 | 28.6 | 0.04 | 3 |
| none | bob | 471 | 259 | 0 (0%) | 212 | 29.4 | 30.0 | 28.8 | 0.06 | 4 |
| none | noisy | 1506 | 72 | 0 (0%) | 1434 | 16.5 | 29.8 | 23.5 | 0.04 | 10 |
| none | **all** | 2412 | 569 | 0 (0%) | 1843 | 29.3 | 30.0 | 28.8 | 0.14 | 18 |
| rpm-200 | alice | 435 | 376 | 0 (0%) | 59 | 28.9 | 30.0 | 28.0 | 0.04 | 3 |
| rpm-200 | bob | 471 | 408 | 0 (0%) | 63 | 29.0 | 30.0 | 28.1 | 0.06 | 4 |
| rpm-200 | noisy | 1506 | 72 | 474 (31%) | 960 | 16.5 | 29.8 | 23.5 | 0.04 | 10 |
| rpm-200 | **all** | 2412 | 856 | 474 (20%) | 1082 | 28.8 | 30.0 | 28.0 | 0.14 | 18 |
| rpm-60 | alice | 435 | 309 | 126 (29%) | 0 | 2.2 | 5.0 | 2.9 | 1.03 | 83 |
| rpm-60 | bob | 471 | 308 | 163 (35%) | 0 | 2.1 | 4.9 | 2.9 | 1.03 | 83 |
| rpm-60 | noisy | 1506 | 309 | 1197 (79%) | 0 | 8.2 | 11.2 | 3.2 | 0.77 | 205 |
| rpm-60 | **all** | 2412 | 926 | 1486 (62%) | 0 | 2.7 | 10.4 | 2.9 | 2.83 | 372 |
| admitq | alice | 435 | 413 | 22 (5%) | 0 | 2.6 | 4.4 | 1.9 | 1.38 | 110 |
| admitq | bob | 471 | 449 | 22 (5%) | 0 | 2.6 | 4.4 | 1.8 | 1.50 | 121 |
| admitq | noisy | 1506 | 400 | 1106 (73%) | 0 | 9.0 | 12.3 | 2.4 | 0.82 | 199 |
| admitq | **all** | 2412 | 1262 | 1150 (48%) | 0 | 3.0 | 12.2 | 2.4 | 3.70 | 430 |
| admitq+share | alice | 435 | 433 | 2 (0%) | 0 | 2.2 | 3.7 | 1.3 | 1.44 | 115 |
| admitq+share | bob | 471 | 471 | 0 (0%) | 0 | 2.2 | 3.6 | 1.4 | 1.57 | 127 |
| admitq+share | noisy | 1506 | 309 | 1197 (79%) | 0 | 9.6 | 12.4 | 2.4 | 0.57 | 132 |
| admitq+share | **all** | 2412 | 1213 | 1199 (50%) | 0 | 2.5 | 12.3 | 2.4 | 3.58 | 374 |

| arm | peak requests waiting inside the engine | requests the engine aborted after a client timeout | engine output |
|---|---:|---:|---:|
| none | 272 | 1843 | 614 tok/s |
| rpm-200 | 194 | 1082 | 639 tok/s |
| rpm-60 | 15 | 0 | 474 tok/s |
| admitq | 0 | 0 | 623 tok/s |
| admitq+share | 0 | 0 | 545 tok/s |

Spread over seeds 0-4 (share of alice's requests that were *good*; min / median / max):

| arm | alice good % | noisy good % |
|---|---|---|
| none | 1.3 / 2.9 / 3.5 | 0.5 / 0.7 / 1.2 |
| rpm-200 | 1.3 / 2.9 / 3.5 | 0.5 / 0.7 / 1.2 |
| rpm-60 | 67.0 / 71.1 / 74.1 | 14.9 / 15.3 / 15.9 |
| admitq | 91.8 / 95.4 / 98.6 | 14.4 / 15.4 / 16.5 |
| admitq+share | 99.5 / 99.8 / 100.0 | 10.1 / 11.3 / 12.3 |

### What it shows

* **No control:** the engine works flat out (614 tokens/s) and almost none of it is
  useful: 76% of requests time out after 30 s, and the ones that complete take 29 s.  Most
  of what the engine generated was for clients that had already left.
* **A request-count limit sized for normal requests (rpm-200)** rejects a third of the
  noisy tenant's requests - and changes nothing, because each one it lets through is as
  heavy as several of alice's.  alice and bob never exceed the limit and still time out.
* **Tightening the count (rpm-60)** does stop the collapse here.  It does so by also
  rejecting 29-35% of alice's and bob's perfectly ordinary traffic, which is over 60/min.
  A count limit can be tuned to protect one traffic mix; it cannot tell an 80-token
  request from an 1800-token one, so there is no setting that is right for both tenants.
* **admitq** keeps the engine's own queue empty.  alice and bob complete 95% of their
  requests with p99 4.4 s; what the noisy tenant gets is whatever is left (about a
  quarter of what it asked for), and the rest is refused in milliseconds with a
  `Retry-After` rather than after 30 s.  alice and bob still lose ~5%: fair queuing decides
  who is next, but it cannot take back slots already held by the noisy tenant's long
  requests.
* **`max_tenant_share=0.5`** closes that gap (alice 99.8% good) at the price of leaving
  capacity idle when only the noisy tenant wants it (545 vs 623 tokens/s of engine
  output).  That trade is yours to make.
* In this run the TPM budget never bites: requests shed for overload are refunded, so the
  noisy tenant never actually spends 200k tokens/minute.  The work is done by the
  in-flight limits, the queue bound and fair queuing.

Limits of the simulation: the fake engine has no KV-cache pressure, no prefix cache and a
fixed speed per sequence; requests within a tenant are similar; estimates are accurate
because prompts are plain ASCII.  Real engines are messier.

## 2. A real engine: llama.cpp on CPU

**REAL, and small.**  One tiny model on a few CPU cores, generating one request at a
time.  No GPU, no continuous batching.  It demonstrates the mechanism end to end on real
software; it says nothing about throughput on real hardware.

```bash
python benchmarks/real_llamacpp.py \
    --server-python /path/to/venv/bin/python \
    --model SmolLM2-135M-Instruct-Q4_K_M.gguf --duration 90 --seed 0
```

**Set-up**

* Model: SmolLM2-135M-Instruct, Q4_K_M GGUF (105 MB).
* Engine: `llama-cpp-python` 0.3.36 server (`python -m llama_cpp.server`), 3 threads,
  `--interrupt_requests false` so that requests queue instead of interrupting each other.
* Proxy: [`examples/fastapi_proxy.py`](../examples/fastapi_proxy.py) under uvicorn, with
  and without the admitq middleware.  Both variants stop the upstream request when the
  client disconnects.
* Load generator: httpx over real sockets, streaming, Poisson arrivals, same process as
  the proxy.
* Machine: a 4-core x86-64 VM that was **also running unrelated CPU-heavy jobs** during
  the runs (load average 9-10).  Single-request speed measured at 20-24 tokens/s at the
  start of each run.  Absolute numbers are therefore noisy and slow; the comparison
  between arms within a run is what matters.

**Traffic (90 s per arm):** alice and bob 0.25 req/s each asking for 16 tokens; noisy 0.8
req/s asking for 96 tokens.  Offered load is 3.5-4.3x what the engine produced.

**admitq configuration:** `max_inflight_requests=2` (one generating, one ready),
`max_inflight_tokens=312`, `max_tenant_share=0.5` (the noisy tenant can never hold both
places), `max_queue_s=6.0`, `cancel_on_disconnect=True`.

### Results

Seed 0:

| arm | tenant | sent | completed | shed | timed out | p50 s | p99 s | good req/s | good tok/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no admission control | alice | 26 | 7 | 0 | 19 | 28.5 | 29.9 | 0.02 | 0.4 |
| no admission control | bob | 21 | 6 | 0 | 15 | 29.2 | 30.0 | 0.01 | 0.2 |
| no admission control | noisy | 95 | 8 | 0 | 87 | 15.5 | 29.8 | 0.02 | 2.1 |
| no admission control | **all** | 142 | 21 | 0 | 121 | 28.5 | 30.0 | 0.06 | 2.7 |
| admitq | alice | 26 | 24 | 2 | 0 | 4.4 | 6.9 | 0.27 | 4.3 |
| admitq | bob | 21 | 21 | 0 | 0 | 4.2 | 5.7 | 0.23 | 3.7 |
| admitq | noisy | 95 | 13 | 82 | 0 | 9.2 | 11.1 | 0.08 | 7.5 |
| admitq | **all** | 142 | 58 | 84 | 0 | 4.6 | 11.1 | 0.58 | 15.5 |

Seed 1:

| arm | tenant | sent | completed | shed | timed out | p50 s | p99 s | good req/s | good tok/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no admission control | alice | 17 | 7 | 0 | 10 | 29.4 | 29.9 | 0.00 | 0.0 |
| no admission control | bob | 24 | 12 | 0 | 12 | 28.5 | 29.8 | 0.04 | 0.7 |
| no admission control | noisy | 83 | 9 | 0 | 74 | 18.1 | 30.0 | 0.02 | 2.1 |
| no admission control | **all** | 124 | 28 | 0 | 96 | 28.5 | 30.0 | 0.07 | 2.8 |
| admitq | alice | 17 | 15 | 2 | 0 | 3.9 | 7.3 | 0.17 | 2.7 |
| admitq | bob | 24 | 22 | 2 | 0 | 4.4 | 6.1 | 0.24 | 3.9 |
| admitq | noisy | 83 | 18 | 65 | 0 | 8.4 | 11.4 | 0.16 | 14.9 |
| admitq | **all** | 124 | 55 | 69 | 0 | 4.9 | 11.4 | 0.57 | 21.5 |

### What it shows - and does not

* Without admission control the queue inside the server grows for the whole run: 77-85%
  of requests time out at 30 s and the median completed request takes 28.5 s.  Goodput is
  0.06-0.07 requests/s.
* With admitq nothing times out.  The quiet tenants complete 82 of 88 requests (93%)
  across the two runs with p99 around 6-7 s; the noisy tenant is refused 78-86% of the
  time, immediately.  Goodput is 0.57-0.58 requests/s, eight to ten times higher, from
  the same engine.
* The quiet tenants' latency is still seconds, not milliseconds: the engine runs one
  generation at a time and cannot be preempted, so a short request arriving behind a
  96-token one waits for it.  Admission control bounds the wait; it does not make a serial
  engine concurrent.
* Usage came from counted stream deltas (`source="counted"`): the requests did not ask
  for `stream_options.include_usage` and `inject_stream_usage` was off.
* Two 90-second runs on a busy shared VM are an existence proof, not a measurement.  The
  counts are small (17-26 requests per quiet tenant per arm), so a single request moves
  those percentages by several points.

**Not tested:** any GPU engine; vLLM, SGLang, TGI or Ollama; the native `llama-server`
binary with parallel slots; continuous batching, where requests slow each other down
gradually instead of queueing strictly; long prompts; multi-worker deployments; the Redis
backend under this load.

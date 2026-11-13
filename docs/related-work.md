# Related work and positioning

Looked at in November 2026.  Descriptions are based on each project's own documentation;
corrections are welcome.

## The problem, in the engines' own trackers

* [vllm-project/vllm#18826](https://github.com/vllm-project/vllm/issues/18826) -
  "[RFC]: Controlling the maximum length of the waiting queue" (open since 2025-05-28).
  The RFC states that vLLM has no mechanism to reject requests based on queue length, that
  the waiting queue is an unbounded deque, and that under overload it grows without limit;
  it proposes `--max-waiting-queue-length` with an HTTP 503.  Since then vLLM has gained
  opt-in global valves, `--max-num-queued-reqs` and `--max-num-queued-tokens` (PR #49445,
  August 2026; see [engines.md](engines.md)); the queue is still unbounded by default.
* [vllm-project/vllm#2901](https://github.com/vllm-project/vllm/issues/2901) -
  "Controlling max queue time" (2024, closed): "Is there a way of controlling when vLLM
  will reject a request because the queue is too long vs let the request enter the queue?"
* [vllm-project/vllm#41250](https://github.com/vllm-project/vllm/issues/41250) -
  "[Feature]: Server overloaded response" (2026, closed): a request for an explicit
  overloaded response for clients.
* [vllm-project/vllm#49538](https://github.com/vllm-project/vllm/issues/49538) -
  "[Feature]: Real-time request-queue signal (NUM_REQUESTS_RUNNING/WAITING) for external
  routers" (open): reports the `/metrics` queue gauges trailing reality by seconds.
* SGLang has a count-based bound, `--max-queued-requests` (see
  [engines.md](engines.md)); issue
  [sgl-project/sglang#45510](https://github.com/sgl-project/sglang/issues/45510) concerns
  its interaction with priority scheduling.
* For llama.cpp no matching issue was found with a brief search; the unbounded deferred
  queue is visible in the source (see [engines.md](engines.md)).

## ASGI rate limiters

[slowapi](https://github.com/laurentS/slowapi),
[fastapi-limiter](https://github.com/long2ice/fastapi-limiter) and
[asgi-ratelimit](https://github.com/abersheeran/asgi-ratelimit) are good at what they are
for: limiting how often a key may call a route.  slowapi adapts flask-limiter's
`"5/minute"` decorators to Starlette; fastapi-limiter is a FastAPI dependency built on
pyrate-limiter (and can block instead of rejecting); asgi-ratelimit is a pure-ASGI
middleware with regex rules, user groups and a Redis backend.

For inference endpoints they leave four things open, all for the same reason - they count
calls and do not look inside them:

1. a 50-token request and a 50 000-token request cost the same;
2. nothing limits how much work is *in flight*, so a burst within the rate still piles up
   inside the engine;
3. there is no queue with a time bound: a request is either inside the rate or rejected;
4. there are no priorities and no fairness between keys beyond each having its own
   counter.

They compose with admitq: keep a request-rate limiter for your non-inference routes (and
as a cheap first filter), and put admitq on the inference routes.

Also related: [pyrate-limiter](https://pypi.org/project/pyrate-limiter/) (weighted items,
several backends) and [aiolimiter](https://pypi.org/project/aiolimiter/) are general
limiters one could build token costing on; [openlimit](https://pypi.org/project/openlimit/)
and [token-throttle](https://pypi.org/project/token-throttle/) limit *outgoing* calls to
hosted APIs by tokens - the client side of the same problem.

## Projects closer to admitq

Found by searching GitHub and PyPI for admission control, token-aware limiting and load
shedding for LLM serving.  None was evaluated beyond its README.

* [mangobanaani/tokenguard](https://github.com/mangobanaani/tokenguard) - a FastAPI rate
  limiter for LLM APIs with hierarchical token buckets (global, team, role, user), Lua
  scripts on Redis, tiktoken estimation and adjustment from actual usage, aimed at
  proxying OpenAI.  The closest match on budgets; it is not about in-flight limits, queue
  bounds or fair queuing.
* [abhijitgunjal/token_guard](https://github.com/abhijitgunjal/token_guard) - token usage
  tracking, policies, budgets and alerts for LLM applications.
* [rockswe/sptc-gate](https://github.com/rockswe/sptc-gate) - a scheduler plugin *inside*
  vLLM/SGLang adding a best-effort class for speculative agent calls.  Complementary: it
  changes what the engine does with admitted work.
* [anujdutt9/multi-tenant-llm-serving](https://github.com/anujdutt9/multi-tenant-llm-serving)
  - a research gateway in front of vLLM with fair queuing and admission control, with
  extensive experiments on an H100.  A study and a gateway rather than a library.
* [harshuljain13/AdmitPerf](https://github.com/harshuljain13/AdmitPerf) - benchmarking of
  admission-control policies for LLM inference.

PyPI name probes on 2026-11-12/13: `admitq`, `llm-admission`, `asgi-admission`,
`inference-admission`, `llm-rate-limiter` and `tokenlimit` were unregistered.

No library was found that packages token-costed in-flight limits, a deadline-bounded fair
queue and usage correction as a drop-in ASGI middleware.  That is a statement about a
brief search, not a proof.

## Gateways - and when to use one instead

admitq is a library that lives inside an app you already run.  A gateway is a service you
run in front of everything.  If you need the service, use the service.

**[LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy)** - an OpenAI-compatible
gateway for 100+ providers with virtual keys, spend tracking, guardrails, load balancing
and an admin dashboard (its documentation also covers per-key rate limits and budgets).  Choose it when you front *many providers or
models*, need key management and cost accounting for teams, or want fallbacks between
deployments.  admitq does none of that.

**[Envoy AI Gateway](https://github.com/envoyproxy/ai-gateway)** (its README now presents
the project as *Agent Router*, formerly Envoy AI Gateway; CRDs and images keep their
names) - a Kubernetes-native control plane on Envoy Gateway: one API for hosted and
self-hosted models, credentials, routing, quotas, failover and usage attribution,
enforced at the proxy.  Choose it when traffic policy belongs to a platform team and
should be enforced outside application code, for every service at once.

**[llm-d](https://llm-d.ai)** - a Kubernetes inference stack around vLLM: load- and
cache-aware routing across replicas, prefill/decode disaggregation, autoscaling, and
"priority- and fairness-aware flow control" for multi-tenant quality of service.  Choose
it when you run *fleets* of model servers; it can make decisions admitq cannot, because
it sees every replica's state.

**[api7/aisix](https://github.com/api7/aisix)** - an open-source AI gateway with
RPS/RPM/TPM limits and concurrency caps per key, model and team (per process, or shared
through Redis), routing and failover, guardrails and caching.

**A gateway is the better choice when** you have more than one application in front of
the models, more than a couple of replicas to route between, a platform team that owns
traffic policy, or requirements around keys, spend and audit.

**admitq fits when** there is one Python service (often one process) in front of one
engine or a few, that service already does your auth and request shaping, and what is
missing is the part between "request arrived" and "engine is busy": pricing the request,
bounding the wait, sharing capacity fairly, and giving the capacity back when a stream
ends.  It is a dependency-free `pip install` and one `add_middleware` call, and it can be
unit-tested with the rest of your app.

They also stack: a gateway for keys and routing in front, admitq in the service next to
each engine as the last, engine-aware line of defence.

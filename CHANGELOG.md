# Changelog

All notable changes to this project are documented here.  The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).  Before 1.0, minor releases
may contain breaking changes; they will be called out.

## [Unreleased]

## [0.1.0] - 2026-11-13

First release.

### Added

- `InferenceAdmission`, a pure-ASGI middleware for inference routes (OpenAI chat,
  completions, embeddings and responses paths by default; custom routes supported).
- `AdmissionController`, the programmatic API (`admit()` / `acquire()` / `Ticket`) for
  use outside ASGI.
- Pre-admission cost estimation from the request body with pluggable estimators:
  `CharEstimator` (default, no dependencies), `TiktokenEstimator`, `TokenizersEstimator`,
  `BodyEstimator` for custom counters.
- Post-hoc correction from real usage: non-streaming `usage`, streamed usage chunks,
  counted SSE deltas; native Ollama and llama.cpp usage fields.
- Admission by in-flight tokens and in-flight requests, with a queue bounded by time
  (`max_queue_s`, client timeout headers), by predicted wait and by depth.
- Shedding with 429 (tenant budgets, exact `Retry-After`) or 503 (overload, estimated
  `Retry-After`) and OpenAI-style error bodies.
- Per-tenant token-per-minute and request-per-minute budgets (token buckets), weighted
  fair queuing between tenants, `max_tenant_share`, priority classes with ageing.
- Budget backends: in-memory and Redis (atomic Lua script, server-side clock).
- Streaming-aware lifecycle: capacity held for the life of the stream and released
  exactly once on completion, error, cancellation or client disconnect; optional
  `cancel_on_disconnect`.
- `AdaptiveLimit`: an AIMD feedback loop on time to first byte or a reported engine
  signal; `admitq.signals` with parsers for vLLM, SGLang, llama.cpp and TGI metrics.
- Metrics: dependency-free registry and Prometheus text endpoint; optional
  `prometheus_client` and OpenTelemetry recorders; tenant label cardinality limit;
  `x-admitq-*` and `x-ratelimit-*` response headers.
- `admitq.testing`: `FakeModelBackend`, `VirtualTimeLoop` / `run_virtual`,
  `simulate_load`, `call_asgi`, `FakeClock`.

[Unreleased]: https://github.com/cb-Sigma/admitq/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cb-Sigma/admitq/releases/tag/v0.1.0

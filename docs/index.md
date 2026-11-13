# admitq documentation

| Page | What is in it |
|---|---|
| [How it works](how-it-works.md) | The five stages a request goes through: estimate, budget, queue, observe, settle.  Fairness, priorities, disconnects, the adaptive limit. |
| [Configuration](configuration.md) | Every option, with defaults.  The programmatic API. |
| [Backends](backends.md) | In-memory versus Redis, and exactly what is and is not consistent across workers. |
| [Metrics](metrics.md) | Metric reference, response headers, Prometheus and OpenTelemetry bindings. |
| [Engine notes](engines.md) | What vLLM, SGLang, llama.cpp, TGI and Ollama do under overload, their queue signals and usage reporting - with sources. |
| [Testing](testing.md) | `admitq.testing`: a fake engine, virtual time and load simulation for tuning settings in unit tests. |
| [Experiments](experiments.md) | The simulated overload experiment and the real llama.cpp one: method, numbers, caveats. |
| [Related work](related-work.md) | Upstream issues, rate limiters, gateways, and when a gateway is the better choice. |

Runnable code is in [`examples/`](../examples) and [`benchmarks/`](../benchmarks).

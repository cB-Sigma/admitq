# Benchmarks

| Script | What it is | Needs |
|---|---|---|
| `simulated_overload.py` | **Simulated** noisy-tenant experiment: fake engine, virtual time, deterministic.  The table in the project README. | nothing but admitq |
| `real_llamacpp.py` | **Real** engine: llama-cpp-python's server on CPU behind `examples/fastapi_proxy.py`, with and without admitq. | `fastapi`, `uvicorn`, `httpx`; an interpreter with `llama-cpp-python[server]`; a small GGUF model |
| `overhead.py` | Microseconds the middleware adds per request and per streamed chunk, in-process. | nothing but admitq |

`results/` holds the raw output of the runs quoted in the documentation:

* `simulated_overload.txt` - `python benchmarks/simulated_overload.py --seeds 5 --markdown`
* `real_llamacpp_seed0.txt`, `real_llamacpp_seed1.txt` -
  `python benchmarks/real_llamacpp.py --model SmolLM2-135M-Instruct-Q4_K_M.gguf --duration 90 --seed N`
  on a 4-core VM that was busy with other work (see the load averages in the files).
  These are the runs quoted in the documentation.
* `real_llamacpp_seed2_30s.txt` - a shorter 30 s run (`--duration 30 --seed 2`) made with
  the code as released, as a sanity check; too short to quote.

Method and interpretation: [docs/experiments.md](../docs/experiments.md).

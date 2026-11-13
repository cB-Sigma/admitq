# Contributing to admitq

Thanks for considering it.  Bug reports with a reproduction, documentation fixes and
focused pull requests are all welcome.

## Set-up

```bash
git clone https://github.com/cb-Sigma/admitq
cd admitq
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The core has no runtime dependencies; `[dev]` pulls in the test and lint tools and the
optional integrations so the whole suite can run.

## The checks

```bash
ruff check .
ruff format --check .
mypy                               # --strict on src/admitq, configured in pyproject.toml
mypy --strict examples benchmarks  # the examples must type-check against real frameworks
pytest
```

CI runs these on Python 3.10 to 3.13, plus the suite with no optional dependencies
installed, the Redis tests against a real server, and a build with a clean install of the
wheel.

Useful variations:

```bash
pytest tests/test_scheduler.py -q            # the deterministic core, < 1 s
HYPOTHESIS_PROFILE=thorough pytest tests/test_scheduler_properties.py tests/test_budgets.py
ADMITQ_REDIS_URL=redis://127.0.0.1:6379/0 pytest tests/test_redis.py -rs
python benchmarks/simulated_overload.py      # the README experiment (simulated)
```

The Redis tests start a private `redis-server` on a free port if the binary is on your
`PATH` and `ADMITQ_REDIS_URL` is not set; otherwise they are skipped.

## How the code is organised

| Module | Role |
|---|---|
| `scheduler.py` | Synchronous state machine: capacity, queue, ordering.  No asyncio, no clock of its own. |
| `controller.py` | Wraps the scheduler with futures, timers, budgets and metrics.  The programmatic API. |
| `middleware.py` | ASGI plumbing: body buffering and replay, response observation, disconnects. |
| `estimators.py`, `usage.py` | Cost before, cost after. |
| `backends/` | Budget storage: memory and Redis, sharing the arithmetic in `base.py`. |
| `adaptive.py`, `signals.py` | The AIMD limit and engine-metric parsing. |
| `metrics.py`, `contrib/` | Registry, Prometheus text, optional bindings. |
| `testing.py` | Public test helpers.  Standard library only. |

## What a good change looks like

* **Tests that pin the behaviour.**  Scheduling, fairness and lifecycle changes need a
  deterministic test (explicit timestamps against `Scheduler`, or virtual time via
  `tests/helpers.virtual`).  If the change touches an invariant - capacity never exceeded,
  released exactly once, queue time bounded, no starvation - extend the property tests.
* **No sleeping in tests.**  Virtual time makes "wait two seconds" free and exact.  The
  only real-time tests are the socket-level ones in `tests/test_server.py` and the Redis
  ones; keep them few and short.
* **The core stays dependency-free.**  New integrations go behind an optional extra and
  are imported lazily or live under `admitq.contrib`.
* **Typed.**  `mypy --strict` must pass; avoid `Any` at public boundaries except where
  ASGI's own looseness requires it (see the note in `middleware.py`).
* **Honest docs.**  If something is estimated, per-process, simulated or untested, the
  docs say so.  Numbers in the README come from scripts in `benchmarks/` that anyone can
  re-run.

## Reporting bugs

A failing test is the best bug report.  `admitq.testing` gives you a fake model server and
an in-process ASGI client, so most problems can be shown in a dozen lines without a GPU:

```python
from admitq import InferenceAdmission
from admitq.testing import FakeModelBackend, call_asgi, run_virtual


async def main():
    app = InferenceAdmission(FakeModelBackend(), max_inflight_requests=1, max_queue_s=0.5)
    response = await call_asgi(app, "POST", "/v1/chat/completions", json_body={"messages": []})
    assert response.status == 200


run_virtual(main())
```

Security problems: please see [SECURITY.md](SECURITY.md) and do not open a public issue.

## Releases

Maintainers: bump `src/admitq/_version.py`, update `CHANGELOG.md`, tag `vX.Y.Z`.  The
release workflow builds and publishes through PyPI trusted publishing.

## Licence

By contributing you agree that your contributions are licensed under the Apache License
2.0, the licence of the project.

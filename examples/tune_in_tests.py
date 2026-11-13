"""Tuning admission settings in a test, with simulated load.

    pip install "admitq @ git+https://github.com/cb-Sigma/admitq" pytest
    pytest examples/tune_in_tests.py -q        # or: python examples/tune_in_tests.py

``admitq.testing`` gives you a fake engine and an event loop with virtual time, so "ten
minutes of overload from three tenants" is a deterministic unit test that runs in a few
seconds.  Use it to answer questions like the ones below before production does.

Model your own engine by changing ``engine()``: how many requests it runs at once and how
many tokens per second it produces.  It is a model - check the conclusions against real
traffic.
"""

from __future__ import annotations

from typing import Any

from admitq import InferenceAdmission, PriorityClass
from admitq.testing import FakeModelBackend, LoadReport, TenantLoad, run_virtual, simulate_load


def engine() -> FakeModelBackend:
    return FakeModelBackend(max_num_seqs=8, total_tps=240.0, stream_tps=30.0)


def tenant(info: Any) -> str:
    return str(info.headers.get("x-tenant-id", "anonymous"))


def protected(**overrides: Any) -> InferenceAdmission:
    settings: dict[str, Any] = {
        "max_inflight_requests": 8,
        "max_inflight_tokens": 6_000,
        "max_queue_s": 2.0,
        "max_tenant_share": 0.5,
        # Batch work may wait ten times longer than interactive work - but still well
        # inside the clients' 60 s timeout, or it would be served only to be thrown away.
        "priorities": [PriorityClass("interactive", 0), PriorityClass("batch", 1, max_queue_s=20)],
        "tenant_key": tenant,
    }
    return InferenceAdmission(engine(), **{**settings, **overrides})


TRAFFIC = [
    TenantLoad("chat-ui", rps=1.0, prompt_tokens=200, max_tokens=60),
    TenantLoad("nightly-eval", rps=6.0, prompt_tokens=600, max_tokens=120, priority="batch"),
]


def run(app: Any, **kwargs: Any) -> LoadReport:
    options: dict[str, Any] = {"duration_s": 300, "slo_s": 8.0, "client_timeout_s": 60.0}
    return run_virtual(simulate_load(app, TRAFFIC, **{**options, **kwargs}))


def test_without_admission_control_the_chat_ui_is_collateral_damage() -> None:
    report = run(engine())
    assert report.tenant("chat-ui").goodput_fraction < 0.2


def test_chat_ui_meets_its_slo_while_the_eval_job_overloads_the_engine() -> None:
    report = run(protected())
    chat = report.tenant("chat-ui")
    assert chat.goodput_fraction > 0.97, report.format()
    assert chat.latency_p99 < 8.0
    # The batch job is shed, not starved: it still gets what capacity is left.
    assert report.tenant("nightly-eval").ok > 100


def test_queue_bound_is_what_clients_see() -> None:
    report = run(protected(max_queue_s=0.5))
    queued = [record.queue_s for record in report.records if record.tenant == "chat-ui"]
    assert max(queued) <= 0.5 + 1e-6


if __name__ == "__main__":
    for name, app in (("no admission control", engine()), ("admitq", protected())):
        print(f"== {name}  (SIMULATED)")
        print(run(app).format())
        print()

"""SIMULATED overload experiment: one noisy tenant, with and without admission control.

Everything here is a simulation: the "engine" is ``admitq.testing.FakeModelBackend`` (a
continuous-batching model with an unbounded internal queue) and time is virtual, so the
run is deterministic and takes seconds.  It shows how the mechanisms interact; it does not
measure any real model server.  See ``benchmarks/real_llamacpp.py`` for a real one.

Run::

    python benchmarks/simulated_overload.py            # table for seed 0
    python benchmarks/simulated_overload.py --seeds 5  # plus the spread over 5 seeds
    python benchmarks/simulated_overload.py --markdown # README-ready tables

Scenario
--------
Engine: 16 concurrent sequences, 800 output tokens/s in total, at most 40 tokens/s per
request, prompts prefilled at 4000 tokens/s.

Traffic (open loop, Poisson, 300 s):

* ``alice`` and ``bob`` - interactive: 1.5 requests/s each, ~150-token prompt, ~80 output
  tokens.  Together they need ~240 output tokens/s: 30% of the engine.
* ``noisy`` - 5 requests/s of ~1500-token prompts producing ~300 output tokens each.
  Alone it asks for ~1500 output tokens/s: almost twice what the engine can produce.

Clients give up after 30 s.  A request counts as *good* if it completes within 10 s.

Arms
----
1. ``none``        - requests go straight to the engine.
2. ``rpm-200``     - a request-count limiter: 200 requests/minute per tenant (what a
                     slowapi / fastapi-limiter style limit expresses).  Emulated with
                     admitq configured with only ``tenant_rpm``.
3. ``rpm-60``      - the same limiter tightened until it bites the noisy tenant.
4. ``admitq``      - token-aware admission: in-flight limits (16 requests, 32 000
                     tokens), 2 s queue bound, 200k TPM per tenant, fair queuing.
5. ``admitq+share`` - the same plus ``max_tenant_share=0.5``: no tenant may hold more
                     than half of the in-flight tokens.

A request-count limit of 200/minute is what an operator would plausibly pick for this
engine if every request looked like alice's (800 tokens/s / 80 tokens = 10 requests/s for
three tenants).  The noisy tenant's 5 requests/s = 300/minute exceeds it, so the limiter
does act - on the count, while each of the requests it lets through is ~4x heavier.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

from admitq import InferenceAdmission
from admitq.testing import FakeModelBackend, LoadReport, TenantLoad, run_virtual, simulate_load

DURATION_S = 300.0
SLO_S = 10.0
CLIENT_TIMEOUT_S = 30.0

TENANTS = [
    TenantLoad("alice", rps=1.5, prompt_tokens=150, max_tokens=120, output_tokens=80),
    TenantLoad("bob", rps=1.5, prompt_tokens=150, max_tokens=120, output_tokens=80),
    TenantLoad("noisy", rps=5.0, prompt_tokens=1500, max_tokens=400, output_tokens=300),
]


def tenant(info: Any) -> str:
    return str(info.headers.get("x-tenant-id", "anonymous"))


def engine() -> FakeModelBackend:
    return FakeModelBackend(max_num_seqs=16, total_tps=800.0, stream_tps=40.0, prefill_tps=4000.0)


def build(arm: str) -> tuple[Any, FakeModelBackend]:
    backend = engine()
    if arm == "none":
        return backend, backend
    if arm.startswith("rpm-"):
        # Request counting only: no token awareness, no queue bound, no in-flight limit.
        rpm = int(arm.split("-")[1])
        return InferenceAdmission(backend, tenant_rpm=rpm, burst_s=10.0, tenant_key=tenant), backend
    if arm in ("admitq", "admitq+share"):
        app = InferenceAdmission(
            backend,
            max_inflight_requests=16,
            max_inflight_tokens=32_000,
            max_queue_s=2.0,
            tenant_tpm=200_000,
            max_tenant_share=0.5 if arm == "admitq+share" else None,
            tenant_key=tenant,
        )
        return app, backend
    raise ValueError(arm)


ARMS = ["none", "rpm-200", "rpm-60", "admitq", "admitq+share"]


def run(arm: str, seed: int) -> tuple[LoadReport, FakeModelBackend]:
    app, backend = build(arm)
    report = run_virtual(
        simulate_load(
            app,
            TENANTS,
            duration_s=DURATION_S,
            seed=seed,
            slo_s=SLO_S,
            client_timeout_s=CLIENT_TIMEOUT_S,
        )
    )
    return report, backend


def fmt(value: float, digits: int = 1) -> str:
    return "-" if value != value else f"{value:.{digits}f}"


def markdown(reports: dict[str, tuple[LoadReport, FakeModelBackend]]) -> str:
    lines = [
        "| arm | tenant | sent | completed | shed | timed out | p50 s | p99 s | "
        "TTFB p99 s | good req/s | good tok/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, (report, _backend) in reports.items():
        for name in [*report.tenants, "all"]:
            s = report.overall if name == "all" else report.tenant(name)
            label = f"**{name}**" if name == "all" else name
            lines.append(
                f"| {arm} | {label} | {s.sent} | {s.ok} | {s.shed} ({100 * s.shed_rate:.0f}%) | "
                f"{s.timeouts} | {fmt(s.latency_p50)} | {fmt(s.latency_p99)} | {fmt(s.ttfb_p99)} | "
                f"{fmt(s.goodput_rps, 2)} | {fmt(s.goodput_tps, 0)} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=0, help="also report the spread over N seeds")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    reports = {arm: run(arm, args.seed) for arm in ARMS}
    print(f"SIMULATED - virtual time, fake engine, seed {args.seed}, {DURATION_S:.0f} s of traffic")
    print(
        f"good = completed within {SLO_S:.0f} s; clients time out after {CLIENT_TIMEOUT_S:.0f} s\n"
    )
    if args.markdown:
        print(markdown(reports))
    else:
        for arm, (report, backend) in reports.items():
            print(f"== {arm}   (engine queue peaked at {backend.peak_waiting} waiting requests)")
            print(report.format())
            print()
    print()
    for arm, (_report, backend) in reports.items():
        print(
            f"{arm:>12}: engine queue peak {backend.peak_waiting:>4}, "
            f"aborted after client timeout {backend.aborted:>4}, "
            f"engine output {backend.generated_tokens / DURATION_S:5.0f} tok/s"
        )

    if args.seeds:
        print(f"\nSpread over seeds 0..{args.seeds - 1} (min / median / max):")
        for arm in ARMS:
            rows = [run(arm, seed)[0] for seed in range(args.seeds)]
            for name in ("alice", "noisy"):
                good = [100 * r.tenant(name).goodput_fraction for r in rows]
                p99 = [r.tenant(name).latency_p99 for r in rows]
                g = f"{min(good):5.1f} / {statistics.median(good):5.1f} / {max(good):5.1f}"
                p = f"{min(p99):6.1f} / {statistics.median(p99):6.1f} / {max(p99):6.1f}"
                print(f"{arm:>12} {name:<6} good% {g}   p99 s {p}")
    print(f"\n(wall time {time.perf_counter() - started:.1f} s)")


if __name__ == "__main__":
    main()

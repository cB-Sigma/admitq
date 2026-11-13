from __future__ import annotations

import re

import pytest

from admitq import METRICS, AdmissionController, Recorder, Registry, Shed, metrics_app
from admitq.testing import call_asgi

from .helpers import virtual

SAMPLE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})? -?[0-9.e+]+(Inf)?$")


def test_registry_counters_gauges_and_histograms():
    registry = Registry()
    assert isinstance(registry, Recorder)
    registry.inc("admitq_admitted_total", {"tenant": "a", "priority": "interactive"})
    registry.inc("admitq_admitted_total", {"tenant": "a", "priority": "interactive"}, 2)
    registry.inc("admitq_admitted_total", {"tenant": "b", "priority": "batch"})
    assert registry.value("admitq_admitted_total", tenant="a", priority="interactive") == 3
    assert registry.total("admitq_admitted_total") == 4
    assert registry.total("admitq_admitted_total", priority="batch") == 1
    registry.set("admitq_queue_depth", {}, 7)
    registry.set("admitq_queue_depth", {}, 3)
    assert registry.value("admitq_queue_depth") == 3
    for value in (0.0005, 0.3, 0.3, 99.0):
        registry.observe("admitq_queue_time_seconds", {"priority": "interactive"}, value)
    assert registry.histogram("admitq_queue_time_seconds", priority="interactive") == (4, 99.6005)
    assert registry.histogram("admitq_queue_time_seconds", priority="none") == (0, 0.0)


def test_prometheus_text_is_well_formed():
    registry = Registry()
    registry.inc(
        "admitq_shed_total", {"tenant": 'we"ird\\ten\nant', "priority": "p", "reason": "r"}
    )
    for value in (0.0005, 0.3, 0.3, 99.0):
        registry.observe("admitq_queue_time_seconds", {"priority": "interactive"}, value)
    registry.set("admitq_inflight_tokens", {}, 1234.5)
    text = registry.render()
    assert text.endswith("\n")
    names = set()
    for line in text.splitlines():
        if line.startswith("# TYPE "):
            _, _, name, kind = line.split()
            names.add(name)
            assert kind == METRICS[name].kind
        elif line.startswith("# HELP "):
            assert line.split()[2] in METRICS
        elif line:
            assert SAMPLE.match(line), line
    assert names == set(METRICS)  # every metric is declared even before it has samples
    assert 'tenant="we\\"ird\\\\ten\\nant"' in text
    assert 'admitq_queue_time_seconds_bucket{priority="interactive",le="0.001"} 1' in text
    assert 'admitq_queue_time_seconds_bucket{priority="interactive",le="0.5"} 3' in text
    assert 'admitq_queue_time_seconds_bucket{priority="interactive",le="+Inf"} 4' in text
    assert 'admitq_queue_time_seconds_count{priority="interactive"} 4' in text
    assert "admitq_inflight_tokens 1234.5" in text
    assert "admitq_queue_depth 0" in text


def test_histogram_buckets_are_cumulative_and_sorted():
    for spec in METRICS.values():
        if spec.kind == "histogram":
            assert list(spec.buckets) == sorted(spec.buckets)
            assert len(spec.buckets) >= 5
        assert spec.name.startswith("admitq_")
        if spec.kind == "counter":
            assert spec.name.endswith("_total")


@virtual
async def test_metrics_app_serves_a_controller_or_registry():
    controller = AdmissionController(max_inflight_tokens=10)
    (await controller.acquire(cost=1)).release()
    for source in (controller, controller.metrics):
        response = await call_asgi(metrics_app(source), "GET", "/")
        assert response.status == 200
        assert "version=0.0.4" in response.headers["content-type"]
        assert b"admitq_admitted_total{" in response.body
        assert int(response.headers["content-length"]) == len(response.body)


async def exercise(controller: AdmissionController) -> None:
    async with controller.admit(tenant="acme", cost=100) as ticket:
        ticket.first_byte()
        ticket.report_usage(prompt_tokens=10, completion_tokens=20)
    with pytest.raises(Shed):
        await controller.acquire(tenant="acme", cost=10_000_000)


@virtual
async def test_prometheus_client_recorder():
    prometheus_client = pytest.importorskip("prometheus_client")
    from admitq.contrib.prometheus import PrometheusRecorder

    registry = prometheus_client.CollectorRegistry()
    recorder = PrometheusRecorder(registry)
    assert isinstance(recorder, Recorder)
    controller = AdmissionController(
        max_inflight_tokens=1000, tenant_tpm=5000, recorders=[recorder]
    )
    await exercise(controller)
    get = registry.get_sample_value
    assert get("admitq_admitted_total", {"tenant": "acme", "priority": "interactive"}) == 1
    assert (
        get(
            "admitq_shed_total",
            {"tenant": "acme", "priority": "interactive", "reason": "tenant_tokens"},
        )
        == 1
    )
    assert get("admitq_completed_total", {"tenant": "acme", "outcome": "ok"}) == 1
    assert get("admitq_actual_tokens_total", {"tenant": "acme", "source": "usage"}) == 30
    assert get("admitq_estimated_tokens_total", {"tenant": "acme"}) == 100
    assert get("admitq_queue_time_seconds_count", {"priority": "interactive"}) == 1
    assert get("admitq_time_to_first_byte_seconds_count") == 1
    assert get("admitq_estimate_ratio_sum", {"source": "usage"}) == pytest.approx(0.3)
    assert get("admitq_inflight_tokens") == 0
    assert get("admitq_inflight_token_limit") == 1000
    # Same numbers as the built-in registry.
    exposition = prometheus_client.generate_latest(registry).decode()
    for name in METRICS:
        assert name in exposition


@virtual
async def test_opentelemetry_recorder():
    pytest.importorskip("opentelemetry.sdk.metrics")
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    from admitq.contrib.opentelemetry import OpenTelemetryRecorder

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    recorder = OpenTelemetryRecorder(provider.get_meter("test"))
    assert isinstance(recorder, Recorder)
    controller = AdmissionController(
        max_inflight_tokens=1000, tenant_tpm=5000, recorders=[recorder]
    )
    await exercise(controller)
    points: dict[str, list] = {}
    units: dict[str, str] = {}
    for resource_metric in reader.get_metrics_data().resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                points[metric.name] = list(metric.data.data_points)
                units[metric.name] = metric.unit
    assert points["admitq.admitted"][0].value == 1
    assert dict(points["admitq.admitted"][0].attributes) == {
        "tenant": "acme",
        "priority": "interactive",
    }
    assert points["admitq.shed"][0].attributes["reason"] == "tenant_tokens"
    assert points["admitq.actual_tokens"][0].value == 30
    assert points["admitq.queue_time"][0].count == 1
    assert points["admitq.inflight_token_limit"][0].value == 1000
    assert points["admitq.inflight_tokens"][0].value == 0
    assert units["admitq.queue_time"] == "s"
    assert units["admitq.actual_tokens"] == "{token}"
    assert units["admitq.admitted"] == "1"

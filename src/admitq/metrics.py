"""Metrics: definitions, a dependency-free registry and a Prometheus text renderer.

The controller reports to any number of :class:`Recorder` objects.  The built-in
:class:`Registry` is one (and is always attached, so ``controller.metrics.render()`` works
out of the box).  ``admitq.contrib.prometheus`` and ``admitq.contrib.opentelemetry``
provide recorders for those libraries.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = ["METRICS", "MetricSpec", "Recorder", "Registry", "metrics_app"]

_QUEUE_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
_TTFT_BUCKETS = (0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
_RATIO_BUCKETS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 4.0, 10.0)


@dataclass(frozen=True, slots=True)
class MetricSpec:
    name: str
    kind: Literal["counter", "gauge", "histogram"]
    help: str
    labels: tuple[str, ...] = ()
    buckets: tuple[float, ...] = ()


_SPECS = (
    MetricSpec(
        "admitq_admitted_total",
        "counter",
        "Requests admitted to the downstream application.",
        ("tenant", "priority"),
    ),
    MetricSpec(
        "admitq_queued_total",
        "counter",
        "Admitted or shed requests that had to wait in the queue first.",
        ("tenant", "priority"),
    ),
    MetricSpec(
        "admitq_shed_total",
        "counter",
        "Requests refused, by reason.",
        ("tenant", "priority", "reason"),
    ),
    MetricSpec(
        "admitq_completed_total",
        "counter",
        "Admitted requests that finished, by outcome.",
        ("tenant", "outcome"),
    ),
    MetricSpec(
        "admitq_queue_time_seconds",
        "histogram",
        "Time admitted requests spent waiting for capacity.",
        ("priority",),
        _QUEUE_BUCKETS,
    ),
    MetricSpec(
        "admitq_time_to_first_byte_seconds",
        "histogram",
        "Time from admission to the first response body byte from the downstream application.",
        (),
        _TTFT_BUCKETS,
    ),
    MetricSpec(
        "admitq_estimate_ratio",
        "histogram",
        "Actual tokens divided by estimated tokens, for requests whose usage was observed.",
        ("source",),
        _RATIO_BUCKETS,
    ),
    MetricSpec(
        "admitq_estimated_tokens_total",
        "counter",
        "Tokens reserved at admission (pre-admission estimates).",
        ("tenant",),
    ),
    MetricSpec(
        "admitq_actual_tokens_total",
        "counter",
        "Tokens charged after correction with observed usage.",
        ("tenant", "source"),
    ),
    MetricSpec("admitq_inflight_tokens", "gauge", "Estimated tokens currently in flight."),
    MetricSpec("admitq_inflight_requests", "gauge", "Requests currently in flight."),
    MetricSpec("admitq_inflight_token_limit", "gauge", "Current in-flight token limit."),
    MetricSpec("admitq_inflight_request_limit", "gauge", "Current in-flight request limit."),
    MetricSpec("admitq_queue_depth", "gauge", "Requests currently waiting for capacity."),
    MetricSpec("admitq_queued_tokens", "gauge", "Estimated tokens currently waiting for capacity."),
    MetricSpec(
        "admitq_backend_errors_total",
        "counter",
        "Budget backend calls that failed or timed out.",
        ("operation",),
    ),
)

#: Every metric admitq emits, by name.
METRICS: Mapping[str, MetricSpec] = {spec.name: spec for spec in _SPECS}


@runtime_checkable
class Recorder(Protocol):
    """Sink for metric updates.  ``labels`` always has exactly the spec's label names."""

    def inc(self, name: str, labels: Mapping[str, str], value: float = 1.0) -> None: ...

    def observe(self, name: str, labels: Mapping[str, str], value: float) -> None: ...

    def set(self, name: str, labels: Mapping[str, str], value: float) -> None: ...


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _fmt(value: float) -> str:
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(float(value))


def _labels(names: tuple[str, ...], values: tuple[str, ...], extra: str = "") -> str:
    parts = [f'{name}="{_escape(value)}"' for name, value in zip(names, values, strict=True)]
    if extra:
        parts.append(extra)
    return "{" + ",".join(parts) + "}" if parts else ""


class Registry:
    """Zero-dependency metric store with Prometheus text exposition."""

    def __init__(self) -> None:
        self._values: dict[str, dict[tuple[str, ...], float]] = {
            spec.name: {} for spec in _SPECS if spec.kind != "histogram"
        }
        # name -> label values -> [bucket counts..., +Inf count, sum]
        self._histograms: dict[str, dict[tuple[str, ...], list[float]]] = {
            spec.name: {} for spec in _SPECS if spec.kind == "histogram"
        }

    @staticmethod
    def _key(spec: MetricSpec, labels: Mapping[str, str]) -> tuple[str, ...]:
        return tuple(labels.get(name, "") for name in spec.labels)

    def inc(self, name: str, labels: Mapping[str, str], value: float = 1.0) -> None:
        series = self._values[name]
        key = self._key(METRICS[name], labels)
        series[key] = series.get(key, 0.0) + value

    def set(self, name: str, labels: Mapping[str, str], value: float) -> None:
        self._values[name][self._key(METRICS[name], labels)] = value

    def observe(self, name: str, labels: Mapping[str, str], value: float) -> None:
        spec = METRICS[name]
        series = self._histograms[name]
        key = self._key(spec, labels)
        data = series.get(key)
        if data is None:
            data = series[key] = [0.0] * (len(spec.buckets) + 2)
        for index, bound in enumerate(spec.buckets):
            if value <= bound:
                data[index] += 1
                break
        else:
            data[len(spec.buckets)] += 1
        data[-1] += value

    # -- reading ----------------------------------------------------------------------

    def value(self, name: str, **labels: str) -> float:
        """Current value of a counter or gauge series (0 if never touched)."""
        return self._values[name].get(self._key(METRICS[name], labels), 0.0)

    def total(self, name: str, **labels: str) -> float:
        """Sum of a counter over every series matching the given labels."""
        spec = METRICS[name]
        total = 0.0
        for key, value in self._values[name].items():
            series = dict(zip(spec.labels, key, strict=True))
            if all(series.get(k) == v for k, v in labels.items()):
                total += value
        return total

    def histogram(self, name: str, **labels: str) -> tuple[int, float]:
        """``(count, sum)`` of a histogram series."""
        data = self._histograms[name].get(self._key(METRICS[name], labels))
        if data is None:
            return 0, 0.0
        return int(sum(data[:-1])), data[-1]

    def render(self) -> str:
        """Prometheus text exposition format (version 0.0.4)."""
        lines: list[str] = []
        for spec in _SPECS:
            lines.append(f"# HELP {spec.name} {spec.help}")
            lines.append(f"# TYPE {spec.name} {spec.kind}")
            if spec.kind == "histogram":
                for key, data in sorted(self._histograms[spec.name].items()):
                    cumulative = 0.0
                    for index, bound in enumerate(spec.buckets):
                        cumulative += data[index]
                        le = f'le="{_fmt(bound)}"'
                        lines.append(
                            f"{spec.name}_bucket{_labels(spec.labels, key, le)} {_fmt(cumulative)}"
                        )
                    cumulative += data[len(spec.buckets)]
                    inf = 'le="+Inf"'
                    lines.append(
                        f"{spec.name}_bucket{_labels(spec.labels, key, inf)} {_fmt(cumulative)}"
                    )
                    lines.append(f"{spec.name}_sum{_labels(spec.labels, key)} {_fmt(data[-1])}")
                    lines.append(f"{spec.name}_count{_labels(spec.labels, key)} {_fmt(cumulative)}")
            else:
                series = self._values[spec.name]
                if not series and not spec.labels:
                    lines.append(f"{spec.name} 0")
                for key, value in sorted(series.items()):
                    lines.append(f"{spec.name}{_labels(spec.labels, key)} {_fmt(value)}")
        lines.append("")
        return "\n".join(lines)


def metrics_app(source: Any) -> Any:
    """A tiny ASGI app serving the Prometheus text format.

    ``source`` is a :class:`Registry`, an :class:`admitq.AdmissionController`, or anything
    else with a ``render()`` method or a ``metrics`` attribute that has one.  Mount it
    wherever your scraper expects it::

        app.mount("/metrics", metrics_app(controller))
    """
    registry = source if hasattr(source, "render") else source.metrics

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            return
        body = registry.render().encode()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"text/plain; version=0.0.4; charset=utf-8"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    return app

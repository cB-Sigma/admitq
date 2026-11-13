"""Report admitq metrics through ``prometheus_client`` (``pip install admitq[prometheus]``).

::

    from admitq.contrib.prometheus import PrometheusRecorder

    app.add_middleware(InferenceAdmission, recorders=[PrometheusRecorder()], ...)

The metrics then appear in whatever ``prometheus_client`` exposition you already have
(``make_asgi_app()``, a push gateway, multiprocess mode, ...), next to your own.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge, Histogram

from ..metrics import METRICS

__all__ = ["PrometheusRecorder"]


class PrometheusRecorder:
    """A :class:`admitq.Recorder` backed by ``prometheus_client`` collectors.

    Create one per registry: registering the same metric names twice in one registry is an
    error in ``prometheus_client``.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        registry = registry if registry is not None else REGISTRY
        self._metrics: dict[str, Any] = {}
        for spec in METRICS.values():
            if spec.kind == "counter":
                # prometheus_client appends "_total" itself.
                name = spec.name.removesuffix("_total")
                self._metrics[spec.name] = Counter(name, spec.help, spec.labels, registry=registry)
            elif spec.kind == "gauge":
                self._metrics[spec.name] = Gauge(
                    spec.name, spec.help, spec.labels, registry=registry
                )
            else:
                self._metrics[spec.name] = Histogram(
                    spec.name, spec.help, spec.labels, buckets=spec.buckets, registry=registry
                )

    def _child(self, name: str, labels: Mapping[str, str]) -> Any:
        metric = self._metrics[name]
        return metric.labels(**labels) if labels else metric

    def inc(self, name: str, labels: Mapping[str, str], value: float = 1.0) -> None:
        self._child(name, labels).inc(value)

    def observe(self, name: str, labels: Mapping[str, str], value: float) -> None:
        self._child(name, labels).observe(value)

    def set(self, name: str, labels: Mapping[str, str], value: float) -> None:
        self._child(name, labels).set(value)

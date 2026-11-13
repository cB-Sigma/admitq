"""Report admitq metrics through OpenTelemetry (``pip install admitq[otel]``).

::

    from admitq.contrib.opentelemetry import OpenTelemetryRecorder

    app.add_middleware(InferenceAdmission, recorders=[OpenTelemetryRecorder()], ...)

Instruments are created on the global meter provider unless you pass a meter.  Metric
names use dots instead of underscores (``admitq.shed``), without the ``_total`` /
``_seconds`` suffixes; units are attached instead.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from opentelemetry import metrics as otel_metrics

from .._version import __version__
from ..metrics import METRICS

__all__ = ["OpenTelemetryRecorder"]


def _otel_name(name: str) -> tuple[str, str]:
    unit = "1"
    base = name.removeprefix("admitq_")
    if base.endswith("_total"):
        base = base.removesuffix("_total")
    if base.endswith("_seconds"):
        base = base.removesuffix("_seconds")
        unit = "s"
    if "tokens" in base:
        unit = "{token}"
    return "admitq." + base, unit


class OpenTelemetryRecorder:
    """A :class:`admitq.Recorder` backed by OpenTelemetry instruments.

    Gauges are exported as observable gauges that report the most recent value.
    """

    def __init__(self, meter: Any | None = None) -> None:
        meter = meter if meter is not None else otel_metrics.get_meter("admitq", __version__)
        self._instruments: dict[str, Any] = {}
        self._gauges: dict[str, dict[tuple[tuple[str, str], ...], float]] = {}
        for spec in METRICS.values():
            name, unit = _otel_name(spec.name)
            if spec.kind == "counter":
                self._instruments[spec.name] = meter.create_counter(
                    name, unit=unit, description=spec.help
                )
            elif spec.kind == "histogram":
                self._instruments[spec.name] = meter.create_histogram(
                    name, unit=unit, description=spec.help
                )
            else:
                values: dict[tuple[tuple[str, str], ...], float] = {}
                self._gauges[spec.name] = values
                meter.create_observable_gauge(
                    name,
                    callbacks=[self._callback(values)],
                    unit=unit,
                    description=spec.help,
                )

    @staticmethod
    def _callback(values: dict[tuple[tuple[str, str], ...], float]) -> Any:
        def observe(_options: Any) -> Iterable[Any]:
            return [
                otel_metrics.Observation(value, dict(labels)) for labels, value in values.items()
            ]

        return observe

    def inc(self, name: str, labels: Mapping[str, str], value: float = 1.0) -> None:
        self._instruments[name].add(value, dict(labels))

    def observe(self, name: str, labels: Mapping[str, str], value: float) -> None:
        self._instruments[name].record(value, dict(labels))

    def set(self, name: str, labels: Mapping[str, str], value: float) -> None:
        self._gauges[name][tuple(sorted(labels.items()))] = value

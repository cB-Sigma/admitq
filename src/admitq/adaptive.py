"""Adaptive in-flight limit: additive increase, multiplicative decrease (AIMD).

The right in-flight limit depends on the model, the hardware, the prompt mix and whatever
else shares the GPU, so a static number is always a guess.  :class:`AdaptiveLimit` moves
the limit from a congestion signal:

* the middleware's own measurement of **time to first byte** (admission to first response
  byte) - available with every engine, no cooperation needed; or
* an **engine signal** you poll yourself and report with
  :meth:`admitq.AdmissionController.report_signal`, such as vLLM's
  ``vllm:num_requests_waiting`` gauge or llama.cpp's ``llamacpp:requests_deferred``.

Once per ``interval_s`` the chosen quantile of the samples collected in that window is
compared with ``target``:

* above target: ``limit <- max(min_limit, limit * decrease)``;
* at or below target *and* the limit was actually being used (peak in-flight at least
  ``utilization`` of the limit): ``limit <- min(max_limit, limit + increase)``;
* otherwise the limit stays where it is - an idle system teaches nothing.

AIMD is the classic choice here because it is stable without a model of the engine: it
probes upward slowly and backs off quickly, and it converges to a band around the point
where the signal crosses the target.  It will oscillate inside that band; that is expected.

Limits of the approach, stated plainly: the signal must actually rise with load (time to
first byte does on every engine we know of, but with a lag of about one request's prefill
time); ``interval_s`` should be several times the typical time to first byte or decisions
are made on requests admitted under the previous limit; and a target set below the
engine's unloaded latency drives the limit to ``min_limit``.
"""

from __future__ import annotations

import math
from typing import Literal

__all__ = ["AdaptiveLimit"]


class AdaptiveLimit:
    """AIMD controller for one in-flight limit.

    Args:
        target: threshold for the signal (seconds for ``signal="ttfb"``, the unit of
            whatever you report for ``signal="reported"``).
        min_limit: the limit never goes below this.
        max_limit: the limit never goes above this.
        initial: starting limit; defaults to the configured static limit.
        dimension: which limit to adapt, ``"tokens"`` or ``"requests"``.  Defaults to
            tokens when ``max_inflight_tokens`` is configured, otherwise requests.
        signal: ``"ttfb"`` to use the measured time to first byte, ``"reported"`` to use
            only values passed to ``report_signal``.
        interval_s: length of one decision window.
        quantile: which quantile of the window's samples is compared with ``target``.
        increase: additive step.  By default the step is one typical request: 1 when
            adapting a request limit, the running mean cost of admitted requests when
            adapting a token limit - "try one more request per window".
        decrease: multiplicative factor in (0, 1) applied when over target.
        min_samples: windows with fewer samples are extended instead of acted on.
        utilization: only increase if peak in-flight reached this fraction of the limit.
    """

    def __init__(
        self,
        *,
        target: float,
        min_limit: int,
        max_limit: int,
        initial: int | None = None,
        dimension: Literal["tokens", "requests"] | None = None,
        signal: Literal["ttfb", "reported"] = "ttfb",
        interval_s: float = 10.0,
        quantile: float = 0.9,
        increase: int | None = None,
        decrease: float = 0.85,
        min_samples: int = 5,
        utilization: float = 0.8,
    ) -> None:
        if target <= 0:
            raise ValueError("target must be positive")
        if min_limit < 1 or max_limit < min_limit:
            raise ValueError("need 1 <= min_limit <= max_limit")
        if not 0.0 < decrease < 1.0:
            raise ValueError("decrease must be in (0, 1)")
        if not 0.0 < quantile <= 1.0:
            raise ValueError("quantile must be in (0, 1]")
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        if initial is not None and not min_limit <= initial <= max_limit:
            raise ValueError("initial must lie within [min_limit, max_limit]")
        self.target = target
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.dimension = dimension
        self.signal = signal
        self.interval_s = interval_s
        self.quantile = quantile
        self.decrease = decrease
        self.min_samples = max(1, min_samples)
        self.utilization = utilization
        self._increase = increase
        self._limit: int | None = initial
        self._samples: list[float] = []
        self._window_start: float | None = None
        self._peak = 0
        self._unit: float | None = None
        #: Number of decreases / increases applied so far (handy in tests and dashboards).
        self.decreases = 0
        self.increases = 0

    @property
    def limit(self) -> int:
        if self._limit is None:
            raise RuntimeError("AdaptiveLimit has not been started")
        return self._limit

    @property
    def increase(self) -> int:
        """The additive step that would be applied now."""
        if self._increase is not None:
            return self._increase
        return max(1, round(self._unit)) if self._unit is not None else 1

    def start(self, configured_limit: int | None, now: float) -> int:
        """Bind to the static configuration.  Returns the starting limit."""
        if self._limit is None:
            base = configured_limit if configured_limit is not None else self.max_limit
            self._limit = min(self.max_limit, max(self.min_limit, base))
        self._window_start = now
        return self._limit

    def note_inflight(self, inflight: int, unit: float = 1.0) -> None:
        """Record an admission: how much of the limit is now in use, and how much of it
        this one request took (``unit``)."""
        if inflight > self._peak:
            self._peak = inflight
        if unit > 0:
            self._unit = unit if self._unit is None else self._unit + 0.05 * (unit - self._unit)

    def record(self, value: float, now: float) -> int | None:
        """Add one signal sample.  Returns the new limit if this sample closed a window
        and the limit changed, else ``None``."""
        if self._limit is None or self._window_start is None:
            raise RuntimeError("AdaptiveLimit has not been started")
        if not math.isfinite(value) or value < 0:
            return None
        if now < self._window_start:
            self._window_start = now  # the clock was swapped (tests, virtual time)
        self._samples.append(value)
        if now - self._window_start < self.interval_s or len(self._samples) < self.min_samples:
            return None

        samples = sorted(self._samples)
        index = min(len(samples) - 1, max(0, math.ceil(self.quantile * len(samples)) - 1))
        observed = samples[index]
        peak = self._peak
        old = self._limit
        self._samples.clear()
        self._peak = 0
        self._window_start = now

        if observed > self.target:
            self._limit = max(self.min_limit, math.floor(old * self.decrease))
            if self._limit < old:
                self.decreases += 1
        elif peak >= self.utilization * old:
            self._limit = min(self.max_limit, old + self.increase)
            if self._limit > old:
                self.increases += 1
        return self._limit if self._limit != old else None

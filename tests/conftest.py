"""Test configuration.

Hypothesis profiles: the default is quick enough for every CI run; set
``HYPOTHESIS_PROFILE=thorough`` for a much deeper search (minutes, not seconds).
"""

from __future__ import annotations

import logging
import os

from hypothesis import HealthCheck, settings

settings.register_profile("default", max_examples=200, deadline=None)
settings.register_profile(
    "thorough",
    max_examples=5000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))

# httpx logs every request at INFO; keep failure output readable.
logging.getLogger("httpx").setLevel(logging.WARNING)

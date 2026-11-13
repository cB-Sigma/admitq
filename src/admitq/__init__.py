"""admitq - token-aware admission control for self-hosted LLM inference endpoints.

Put it in front of the model server you already run (vLLM, llama.cpp, SGLang, TGI,
Ollama) inside the ASGI app you already have::

    from admitq import InferenceAdmission

    app.add_middleware(
        InferenceAdmission,
        max_queue_s=2.0,
        max_inflight_tokens=64_000,
        tenant_tpm={"default": 200_000},
    )

or use :class:`AdmissionController` directly anywhere you can ``await``.
"""

from ._version import __version__
from .adaptive import AdaptiveLimit
from .backends import BudgetBackend, MemoryBackend, Reservation, TenantLimits
from .config import AdmissionConfig, PriorityClass
from .controller import AdmissionController, Ticket
from .errors import AdmitqError, ConfigError, Shed, ShedReason
from .estimators import (
    BodyEstimator,
    CharEstimator,
    Cost,
    CostEstimator,
    TiktokenEstimator,
    TokenizersEstimator,
)
from .metrics import METRICS, Recorder, Registry, metrics_app
from .middleware import InferenceAdmission
from .request import (
    RequestInfo,
    api_key_tenant,
    body_field_tenant,
    header_priority,
    header_tenant,
    jwt_claim_tenant,
)
from .routes import DEFAULT_ROUTES, RouteMatcher
from .scheduler import Scheduler, SchedulerSnapshot
from .usage import ObservedUsage, ResponseObserver, extract_usage

__all__ = [
    "DEFAULT_ROUTES",
    "METRICS",
    "AdaptiveLimit",
    "AdmissionConfig",
    "AdmissionController",
    "AdmitqError",
    "BodyEstimator",
    "BudgetBackend",
    "CharEstimator",
    "ConfigError",
    "Cost",
    "CostEstimator",
    "InferenceAdmission",
    "MemoryBackend",
    "ObservedUsage",
    "PriorityClass",
    "Recorder",
    "Registry",
    "RequestInfo",
    "Reservation",
    "ResponseObserver",
    "RouteMatcher",
    "Scheduler",
    "SchedulerSnapshot",
    "Shed",
    "ShedReason",
    "TenantLimits",
    "Ticket",
    "TiktokenEstimator",
    "TokenizersEstimator",
    "__version__",
    "api_key_tenant",
    "body_field_tenant",
    "extract_usage",
    "header_priority",
    "header_tenant",
    "jwt_claim_tenant",
    "metrics_app",
]

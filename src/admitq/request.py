"""What admitq knows about a request, and functions that derive tenant and priority."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "RequestInfo",
    "api_key_tenant",
    "body_field_tenant",
    "header_priority",
    "header_tenant",
    "jwt_claim_tenant",
]


@dataclass(frozen=True, slots=True)
class RequestInfo:
    """A request as seen by tenant/priority key functions.

    ``headers`` has lower-case names.  ``body`` is the parsed JSON object, or ``None`` when
    the body was not JSON, not an object, or too large to buffer.
    """

    method: str
    path: str
    endpoint: str
    headers: Mapping[str, str]
    body: Mapping[str, Any] | None = None
    scope: Mapping[str, Any] = field(default_factory=dict)


KeyFunction = Callable[[RequestInfo], "str | None"]


def _bearer(info: RequestInfo) -> str | None:
    value = info.headers.get("authorization", "")
    scheme, _, token = value.partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    return info.headers.get("x-api-key") or info.headers.get("api-key") or None


def api_key_tenant(*, default: str = "anonymous") -> KeyFunction:
    """Tenant = the API key (``Authorization: Bearer ...``, ``x-api-key`` or ``api-key``).

    The key itself never becomes the tenant name: a short SHA-256 prefix is used, so keys
    do not leak into metrics labels, logs or Redis key names.
    """

    def key(info: RequestInfo) -> str:
        token = _bearer(info)
        if not token:
            return default
        return "key-" + hashlib.sha256(token.encode()).hexdigest()[:12]

    return key


def header_tenant(name: str = "x-tenant-id", *, default: str = "anonymous") -> KeyFunction:
    """Tenant = the value of a header.  Only use a header your own edge sets or verifies;
    a client that can choose its tenant can choose somebody else's budget."""
    lowered = name.lower()

    def key(info: RequestInfo) -> str:
        return info.headers.get(lowered) or default

    return key


def body_field_tenant(name: str = "user", *, default: str = "anonymous") -> KeyFunction:
    """Tenant = a top-level string field of the JSON body (OpenAI's ``user`` by default)."""

    def key(info: RequestInfo) -> str:
        if info.body is not None:
            value = info.body.get(name)
            if isinstance(value, str) and value:
                return value
        return default

    return key


def jwt_claim_tenant(claim: str = "sub", *, default: str = "anonymous") -> KeyFunction:
    """Tenant = a claim of the bearer JWT.

    The token is **decoded, not verified**.  That is only safe behind something that has
    already verified the signature (your auth middleware or gateway).  If you verify in
    the application, put that middleware outside admitq and read the verified identity
    from ``info.scope`` in your own key function instead.
    """

    def key(info: RequestInfo) -> str:
        token = _bearer(info)
        if not token:
            return default
        parts = token.split(".")
        if len(parts) != 3:
            return default
        payload = parts[1]
        try:
            decoded = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
            claims = json.loads(decoded)
        except (ValueError, binascii.Error):
            return default
        value = claims.get(claim) if isinstance(claims, dict) else None
        if isinstance(value, str | int) and not isinstance(value, bool) and str(value):
            return str(value)
        return default

    return key


def header_priority(name: str = "x-priority") -> KeyFunction:
    """Priority = the value of a header (unknown or missing values get the default class).

    Clients choose this themselves, so treat it as a hint between cooperating callers.  If
    priorities must be enforced, derive them from the authenticated identity instead.
    """
    lowered = name.lower()

    def key(info: RequestInfo) -> str | None:
        return info.headers.get(lowered) or None

    return key

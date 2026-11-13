"""Which requests are inference requests.

Everything that does not match is passed straight through: the middleware does not read
its body, does not time it and does not count it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

__all__ = ["DEFAULT_ROUTES", "RouteMatcher"]

#: Path suffix -> endpoint kind.  Matching is by suffix so the same table works when the
#: app is mounted under a prefix (``/openai/v1/chat/completions``).
DEFAULT_ROUTES: Mapping[str, str] = {
    "/v1/chat/completions": "chat",
    "/v1/completions": "completions",
    "/v1/embeddings": "embeddings",
    "/v1/responses": "responses",
}

_KINDS = frozenset({"chat", "completions", "embeddings", "responses", "other"})


class RouteMatcher:
    """Maps ``(method, path)`` to an endpoint kind, or ``None`` for "not ours".

    Args:
        routes: ``{path: kind}``.  A path matches if the request path equals it or ends
            with it; a key starting with ``^`` is a regular expression matched against the
            whole path.  ``kind`` is one of ``chat``, ``completions``, ``embeddings``,
            ``responses`` or ``other`` and tells the estimator how to read the body
            (``embeddings`` reserves no output tokens).
        methods: HTTP methods considered at all.
    """

    def __init__(
        self,
        routes: Mapping[str, str] | Iterable[str] | None = None,
        *,
        methods: Iterable[str] = ("POST",),
    ) -> None:
        if routes is None:
            table = dict(DEFAULT_ROUTES)
        elif isinstance(routes, Mapping):
            table = dict(routes)
        else:
            table = {path: DEFAULT_ROUTES.get(path, "other") for path in routes}
        self._exact: dict[str, str] = {}
        self._suffixes: list[tuple[str, str]] = []
        self._patterns: list[tuple[re.Pattern[str], str]] = []
        for path, kind in table.items():
            if kind not in _KINDS:
                raise ValueError(f"unknown endpoint kind {kind!r} for route {path!r}")
            if path.startswith("^"):
                self._patterns.append((re.compile(path), kind))
            else:
                if not path.startswith("/"):
                    raise ValueError(f"route {path!r} must start with '/' (or '^' for a regex)")
                normalized = path.rstrip("/") or "/"
                self._exact[normalized] = kind
                self._suffixes.append((normalized, kind))
        # Longest suffix first so "/v1/chat/completions" wins over "/completions".
        self._suffixes.sort(key=lambda item: len(item[0]), reverse=True)
        self._methods = frozenset(method.upper() for method in methods)

    def match(self, method: str, path: str) -> str | None:
        if method.upper() not in self._methods:
            return None
        if len(path) > 1 and path.endswith("/"):
            path = path.rstrip("/")
        kind = self._exact.get(path)
        if kind is not None:
            return kind
        for suffix, kind in self._suffixes:
            if path.endswith(suffix):
                return kind
        for pattern, kind in self._patterns:
            if pattern.fullmatch(path):
                return kind
        return None

"""Shared test helpers."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable, Coroutine
from typing import Any

from admitq.testing import run_virtual


def virtual(fn: Callable[..., Coroutine[Any, Any, None]]) -> Callable[..., None]:
    """Run an async test on a virtual-time event loop (instant and deterministic)."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        run_virtual(fn(*args, **kwargs))

    return wrapper


async def settle(rounds: int = 5) -> None:
    """Let ready callbacks and tasks run without advancing virtual time."""
    for _ in range(rounds):
        await asyncio.sleep(0)


def tenant_header(info: Any) -> str:
    return info.headers.get("x-tenant-id", "anonymous")

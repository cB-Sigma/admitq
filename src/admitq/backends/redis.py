"""Redis budget backend: per-tenant budgets shared by every worker and replica.

Each tenant's two buckets live in one hash; refill, check and debit happen inside a single
Lua script, so concurrent workers cannot both spend the same budget.  Time comes from the
Redis server (``TIME``), so worker clock skew does not matter.

What is shared: tenant token-per-minute and request-per-minute budgets.
What is **not** shared: the in-flight limits and the wait queue.  Those are held by each
process's scheduler, because waking a waiter in another process cannot be done cheaply or
exactly.  Give each worker its share of the engine's capacity.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .base import Reservation, TenantLimits

__all__ = ["RedisBackend"]

# KEYS[1] = bucket hash
# ARGV = mode ("reserve" | "adjust"), tokens, requests,
#        tok_rate, tok_cap, req_rate, req_cap, ttl_ms
_LUA = """
local mode = ARGV[1]
local tokens = tonumber(ARGV[2])
local requests = tonumber(ARGV[3])
local tok_rate = tonumber(ARGV[4])
local tok_cap = tonumber(ARGV[5])
local req_rate = tonumber(ARGV[6])
local req_cap = tonumber(ARGV[7])
local ttl_ms = tonumber(ARGV[8])

local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000

local h = redis.call('HMGET', KEYS[1], 'tok', 'req', 'ts')
local tok = tonumber(h[1])
local req = tonumber(h[2])
local ts = tonumber(h[3])
if tok == nil or req == nil or ts == nil then
  tok = tok_cap
  req = req_cap
  ts = now
end
local elapsed = now - ts
if elapsed < 0 then elapsed = 0 end
if tok_rate > 0 then tok = math.min(tok_cap, tok + elapsed * tok_rate) end
if req_rate > 0 then req = math.min(req_cap, req + elapsed * req_rate) end

local allowed = 1
local reason = ''
local retry = 0
if mode == 'reserve' then
  local wait_tok = 0
  local wait_req = 0
  if tok_rate > 0 then
    local need = math.min(tokens, tok_cap)
    if tok < need then wait_tok = (need - tok) / tok_rate end
  end
  if req_rate > 0 then
    local need = math.min(requests, req_cap)
    if req < need then wait_req = (need - req) / req_rate end
  end
  if wait_tok > 0 or wait_req > 0 then
    allowed = 0
    if wait_tok >= wait_req then
      reason = 'tenant_tokens'
      retry = wait_tok
    else
      reason = 'tenant_requests'
      retry = wait_req
    end
  else
    if tok_rate > 0 then tok = tok - tokens end
    if req_rate > 0 then req = req - requests end
  end
else
  if tok_rate > 0 then tok = math.max(-tok_cap, math.min(tok_cap, tok - tokens)) end
  if req_rate > 0 then req = math.max(-req_cap, math.min(req_cap, req - requests)) end
end

redis.call('HSET', KEYS[1], 'tok', tostring(tok), 'req', tostring(req), 'ts', tostring(now))
redis.call('PEXPIRE', KEYS[1], ttl_ms)
return {allowed, reason, tostring(retry), tostring(tok), tostring(req)}
"""


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes | bytearray) else str(value)


class RedisBackend:
    """Budgets in Redis (``pip install admitq[redis]``).

    Args:
        client: a ``redis.asyncio.Redis`` (or ``RedisCluster``) instance.  The backend does
            not close a client it was handed unless ``owns_client`` is true.
        prefix: key prefix.  Keys look like ``admitq:{tenant}`` - the braces are a cluster
            hash tag, so each script call touches exactly one slot.
        timeout: seconds to wait for Redis before treating the call as failed.  What a
            failure means for the request is decided by the controller's
            ``on_backend_error`` setting (fail open by default).
    """

    def __init__(
        self,
        client: Any,
        *,
        prefix: str = "admitq",
        timeout: float = 0.25,
        owns_client: bool = False,
    ) -> None:
        self._client = client
        self._prefix = prefix
        self._timeout = timeout
        self._owns_client = owns_client
        self._script = client.register_script(_LUA)

    @classmethod
    def from_url(cls, url: str, *, prefix: str = "admitq", timeout: float = 0.25) -> RedisBackend:
        try:
            from redis.asyncio import Redis
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError("RedisBackend needs redis: pip install 'admitq[redis]'") from exc
        return cls(Redis.from_url(url), prefix=prefix, timeout=timeout, owns_client=True)

    def _key(self, tenant: str) -> str:
        return f"{self._prefix}:{{{tenant}}}"

    async def _run(
        self, mode: str, tenant: str, tokens: int, requests: int, limits: TenantLimits
    ) -> list[Any]:
        ttl_ms = int(max(1.0, 2.0 * limits.burst_s) * 1000)
        args = [
            mode,
            tokens,
            requests,
            repr(limits.token_rate),
            repr(limits.token_capacity),
            repr(limits.request_rate),
            repr(limits.request_capacity),
            ttl_ms,
        ]
        result = await asyncio.wait_for(
            self._script(keys=[self._key(tenant)], args=args), self._timeout
        )
        return list(result)

    async def reserve(
        self, tenant: str, tokens: int, requests: int, limits: TenantLimits
    ) -> Reservation:
        allowed, reason, retry, tok, req = await self._run(
            "reserve", tenant, tokens, requests, limits
        )
        remaining_tokens = max(0.0, float(_text(tok))) if limits.tpm else None
        remaining_requests = max(0.0, float(_text(req))) if limits.rpm else None
        if int(allowed):
            return Reservation(
                allowed=True,
                remaining_tokens=remaining_tokens,
                remaining_requests=remaining_requests,
            )
        why = _text(reason)
        return Reservation(
            allowed=False,
            reason="tenant_requests" if why == "tenant_requests" else "tenant_tokens",
            retry_after=float(_text(retry)),
            remaining_tokens=remaining_tokens,
            remaining_requests=remaining_requests,
        )

    async def adjust(self, tenant: str, tokens: int, requests: int, limits: TenantLimits) -> None:
        await self._run("adjust", tenant, tokens, requests, limits)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

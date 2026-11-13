from __future__ import annotations

import base64
import json

from admitq import (
    RequestInfo,
    api_key_tenant,
    body_field_tenant,
    header_priority,
    header_tenant,
    jwt_claim_tenant,
)
from admitq.signals import (
    llamacpp_requests_deferred,
    prometheus_value,
    sglang_queue_requests,
    tgi_queue_size,
    vllm_requests_waiting,
)


def info(headers=None, body=None) -> RequestInfo:
    return RequestInfo("POST", "/v1/chat/completions", "chat", headers or {}, body)


def jwt(claims: dict) -> str:
    def part(obj) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    return f"{part({'alg': 'none'})}.{part(claims)}.signature"


def test_api_key_tenant_hashes_the_key():
    key = api_key_tenant()
    a = key(info({"authorization": "Bearer sk-abc"}))
    assert a.startswith("key-")
    assert len(a) == 16
    assert "sk-abc" not in a
    assert key(info({"authorization": "bearer sk-abc"})) == a  # scheme is case-insensitive
    assert key(info({"authorization": "Bearer sk-other"})) != a
    assert key(info({"x-api-key": "sk-abc"})) == a
    assert key(info({"api-key": "sk-abc"})) == a  # Azure-style header
    assert key(info({})) == "anonymous"
    assert key(info({"authorization": "Basic dXNlcjpwYXNz"})) == "anonymous"
    assert key(info({"authorization": "Bearer "})) == "anonymous"
    assert api_key_tenant(default="public")(info({})) == "public"


def test_header_and_body_tenants():
    assert header_tenant("X-Org-ID")(info({"x-org-id": "acme"})) == "acme"
    assert header_tenant()(info({})) == "anonymous"
    assert header_tenant(default="nobody")(info({"x-tenant-id": ""})) == "nobody"
    assert body_field_tenant()(info(body={"user": "u-1"})) == "u-1"
    assert body_field_tenant("org")(info(body={"org": 5})) == "anonymous"
    assert body_field_tenant()(info(body=None)) == "anonymous"


def test_jwt_claim_tenant_decodes_without_verifying():
    key = jwt_claim_tenant("org_id")
    assert key(info({"authorization": f"Bearer {jwt({'org_id': 'acme', 'sub': 'u1'})}"})) == "acme"
    assert jwt_claim_tenant()(info({"authorization": f"Bearer {jwt({'sub': 42})}"})) == "42"
    for bad in (
        "Bearer not-a-jwt",
        "Bearer a.b.c",
        "Bearer a.!!!.c",
        f"Bearer {jwt({'other': 'x'})}",
        f"Bearer {jwt({'org_id': True})}",
        f"Bearer {jwt({'org_id': ''})}",
        "Bearer e30.WzEsMl0.sig",  # payload is a JSON list
    ):
        assert key(info({"authorization": bad})) == "anonymous", bad


def test_header_priority():
    assert header_priority()(info({"x-priority": "batch"})) == "batch"
    assert header_priority("X-Tier")(info({"x-tier": "gold"})) == "gold"
    assert header_priority()(info({})) is None
    assert header_priority()(info({"x-priority": ""})) is None


VLLM = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} 12.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="m"} 37.0
vllm:num_requests_waiting{engine="1",model_name="m"} 3.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="m",reason="capacity"} 99.0
"""


def test_prometheus_value_parsing():
    assert vllm_requests_waiting(VLLM) == 40.0  # summed over label sets, not the _by_reason one
    assert prometheus_value(VLLM, "vllm:num_requests_running") == 12.0
    assert prometheus_value(VLLM, "vllm:missing") is None
    assert prometheus_value("x 1 1700000000000\n", "x") == 1.0  # trailing timestamp
    assert prometheus_value('x{a="}"} 2\n', "x") == 2.0
    assert prometheus_value("x NaN-ish\nx{broken 3\n", "x") is None
    assert llamacpp_requests_deferred("llamacpp:requests_deferred 5\n") == 5.0
    assert sglang_queue_requests('sglang:num_queue_reqs{model_name="m"} 7.0\n') == 7.0
    assert tgi_queue_size("tgi_queue_size 2\ntgi_queue_size_other 9\n") == 2.0

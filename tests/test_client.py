from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

import stabilizer_python_sdk.client as client_module
from stabilizer_python_sdk import (
    ApiError,
    ResponseEnvelope,
    StabilizerClient,
)


@dataclass
class RecordedRequest:
    method: str
    url: str
    headers: dict[str, str]
    json_body: object | None
    timeout: float


class FakeTransport:
    def __init__(self, responses: list[ResponseEnvelope]) -> None:
        self._responses: Iterator[ResponseEnvelope] = iter(responses)
        self.requests: list[RecordedRequest] = []

    def send(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        json_body: object | None,
        timeout: float,
    ) -> ResponseEnvelope:
        self.requests.append(
            RecordedRequest(
                method=method,
                url=url,
                headers=headers,
                json_body=json_body,
                timeout=timeout,
            )
        )
        return next(self._responses)


def test_health_uses_public_base_url_without_auth_header() -> None:
    transport = FakeTransport(
        [ResponseEnvelope(status_code=200, data={"status": "ok", "version": "v1"})]
    )
    client = StabilizerClient(transport=transport)

    response = client.health()

    assert response == {"status": "ok", "version": "v1"}
    assert transport.requests == [
        RecordedRequest(
            method="GET",
            url="https://stabilizerapi.documentinsight.ai/api/v1/health",
            headers={},
            json_body=None,
            timeout=30.0,
        )
    ]


def test_public_routes_skip_auth_even_with_api_key_when_path_needs_normalization() -> None:
    client = StabilizerClient(api_key="sk_test")

    assert client._auth_headers("/v1/health/") == {}
    assert client._auth_headers("/v1/supported-models?format=json") == {}


def test_non_public_get_routes_include_bearer_auth_without_request_body() -> None:
    transport = FakeTransport([ResponseEnvelope(status_code=200, data={"org_id": "org_123"})])
    client = StabilizerClient(api_key="sk_test", transport=transport)

    response = client.get_org()

    assert response == {"org_id": "org_123"}
    assert transport.requests == [
        RecordedRequest(
            method="GET",
            url="https://stabilizerapi.documentinsight.ai/api/v1/org",
            headers={"Authorization": "Bearer sk_test"},
            json_body=None,
            timeout=30.0,
        )
    ]


def test_walkthrough_flow_posts_expected_payloads_with_bearer_auth() -> None:
    transport = FakeTransport(
        [
            ResponseEnvelope(status_code=201, data={"config_id": "cfg_123"}),
            ResponseEnvelope(status_code=202, data={"job_id": "job_opt", "status": "queued"}),
            ResponseEnvelope(status_code=202, data={"job_id": "job_compile", "status": "queued"}),
            ResponseEnvelope(status_code=202, data={"job_id": "job_extract", "status": "queued"}),
        ]
    )
    client = StabilizerClient(api_key="sk_test", transport=transport)

    llm_payload = {
        "name": "Primary config",
        "provider": "openai",
        "api_key": "provider-key",
        "default_model": "google/gemini-2.5-flash-lite",
        "is_default": True,
    }
    optimize_payload = {
        "prompt": "Extract fields",
        "json_structure": {"event_title": "string"},
        "training_data": [{"source_text": "x", "extracted_json": {"event_title": "y"}}] * 5,
    }
    compile_payload = {
        "name": "Event details extractor",
        "description": "Extracts event details from text",
        "tags": ["events", "walkthrough"],
        "prompt": "Extract event details",
        "json_structure": {"event_title": "string"},
        "training_data": [{"source_text": "x", "extracted_json": {"event_title": "y"}}],
        "grounding_methods": ["hard_grounding", "constraints_validation"],
        "compile_options": {"num_prompt_variations": 3},
    }
    extract_payload = {
        "function_id": "fn_123",
        "source_text": "Harbor Lights Food Fair...",
        "options": {"num_results": 3},
    }

    assert client.create_llm_config(llm_payload) == {"config_id": "cfg_123"}
    assert client.optimize_prompt(optimize_payload)["job_id"] == "job_opt"
    assert client.compile_function(compile_payload)["job_id"] == "job_compile"
    assert client.extract(extract_payload)["job_id"] == "job_extract"

    assert [request.method for request in transport.requests] == ["POST", "POST", "POST", "POST"]
    assert [request.url for request in transport.requests] == [
        "https://stabilizerapi.documentinsight.ai/api/v1/llm-configs",
        "https://stabilizerapi.documentinsight.ai/api/v1/prompt-optimizations",
        "https://stabilizerapi.documentinsight.ai/api/v1/functions",
        "https://stabilizerapi.documentinsight.ai/api/v1/extract",
    ]
    for request in transport.requests:
        assert request.headers == {
            "Authorization": "Bearer sk_test",
            "Content-Type": "application/json",
        }
    assert transport.requests[0].json_body == llm_payload
    assert transport.requests[1].json_body == optimize_payload
    assert transport.requests[2].json_body == compile_payload
    assert transport.requests[3].json_body == extract_payload


def test_get_api_key_uses_key_scoped_path_with_bearer_auth() -> None:
    transport = FakeTransport(
        [ResponseEnvelope(status_code=200, data={"key_id": "key_123", "revoked": False})]
    )
    client = StabilizerClient(api_key="sk_test", transport=transport)

    response = client.get_api_key("key_123")

    assert response == {"key_id": "key_123", "revoked": False}
    assert transport.requests == [
        RecordedRequest(
            method="GET",
            url="https://stabilizerapi.documentinsight.ai/api/v1/api-keys/key_123",
            headers={"Authorization": "Bearer sk_test"},
            json_body=None,
            timeout=30.0,
        )
    ]


def test_get_api_key_accepts_me_alias() -> None:
    transport = FakeTransport([ResponseEnvelope(status_code=200, data={"key_id": "key_self"})])
    client = StabilizerClient(api_key="sk_test", transport=transport)

    client.get_api_key("me")

    assert transport.requests[0].url == "https://stabilizerapi.documentinsight.ai/api/v1/api-keys/me"


def test_update_api_key_patches_with_json_body() -> None:
    transport = FakeTransport(
        [ResponseEnvelope(status_code=200, data={"key_id": "key_123", "revoked": True})]
    )
    client = StabilizerClient(api_key="sk_test", transport=transport)

    payload = {"budget": {"extract_limit": 0, "period": "month"}, "revoked": True}
    response = client.update_api_key("key_123", payload)

    assert response == {"key_id": "key_123", "revoked": True}
    assert transport.requests == [
        RecordedRequest(
            method="PATCH",
            url="https://stabilizerapi.documentinsight.ai/api/v1/api-keys/key_123",
            headers={"Authorization": "Bearer sk_test", "Content-Type": "application/json"},
            json_body=payload,
            timeout=30.0,
        )
    ]


def test_get_api_key_usage_forwards_from_and_to_query() -> None:
    transport = FakeTransport(
        [ResponseEnvelope(status_code=200, data={"key_id": "key_123", "extract_count": 4})]
    )
    client = StabilizerClient(api_key="sk_test", transport=transport)

    response = client.get_api_key_usage("key_123", from_="2026-04-01", to="2026-04-15")

    assert response == {"key_id": "key_123", "extract_count": 4}
    assert transport.requests[0].method == "GET"
    assert transport.requests[0].url == (
        "https://stabilizerapi.documentinsight.ai/api/v1/api-keys/key_123/usage"
        "?from=2026-04-01&to=2026-04-15"
    )


def test_get_api_key_usage_omits_absent_query_params() -> None:
    transport = FakeTransport([ResponseEnvelope(status_code=200, data={"key_id": "key_123"})])
    client = StabilizerClient(api_key="sk_test", transport=transport)

    client.get_api_key_usage("key_123")

    assert transport.requests[0].url == (
        "https://stabilizerapi.documentinsight.ai/api/v1/api-keys/key_123/usage"
    )


def test_test_llm_config_posts_provider_key_verification_body() -> None:
    transport = FakeTransport([ResponseEnvelope(status_code=200, data={"valid": True, "reason": "ok"})])
    client = StabilizerClient(api_key="sk_test", transport=transport)

    payload = {"api_key": "provider-key", "default_model": "google/gemini-2.5-flash-lite"}
    response = client.test_llm_config(payload)

    assert response == {"valid": True, "reason": "ok"}
    assert transport.requests == [
        RecordedRequest(
            method="POST",
            url="https://stabilizerapi.documentinsight.ai/api/v1/llm-configs/test",
            headers={"Authorization": "Bearer sk_test", "Content-Type": "application/json"},
            json_body=payload,
            timeout=30.0,
        )
    ]


def test_get_usage_forwards_group_by_limit_and_cursor() -> None:
    transport = FakeTransport([ResponseEnvelope(status_code=200, data={"keys": [], "next_cursor": None})])
    client = StabilizerClient(api_key="sk_test", transport=transport)

    client.get_usage(from_="2026-04-01", to="2026-04-15", group_by="key", limit=50, cursor="key_123")

    assert transport.requests[0].url == (
        "https://stabilizerapi.documentinsight.ai/api/v1/usage"
        "?from=2026-04-01&to=2026-04-15&group_by=key&limit=50&cursor=key_123"
    )


def test_get_usage_without_grouping_only_sends_date_window() -> None:
    transport = FakeTransport([ResponseEnvelope(status_code=200, data={"total_jobs": 7})])
    client = StabilizerClient(api_key="sk_test", transport=transport)

    client.get_usage(from_="2026-04-01", to="2026-04-15")

    assert transport.requests[0].url == (
        "https://stabilizerapi.documentinsight.ai/api/v1/usage?from=2026-04-01&to=2026-04-15"
    )


def test_stabilizer_client_does_not_expose_wait_for_job() -> None:
    client = StabilizerClient(api_key="sk_test", transport=FakeTransport([]))

    assert not hasattr(client, "wait_for_job")


def test_non_successful_response_raises_api_error_with_status_and_payload() -> None:
    transport = FakeTransport(
        [
            ResponseEnvelope(
                status_code=404,
                data={"error": {"code": "not_found", "message": "Function not found"}},
            )
        ]
    )
    client = StabilizerClient(api_key="sk_test", transport=transport)

    with pytest.raises(ApiError) as exc_info:
        client.get_function("fn_missing")

    assert exc_info.value.status_code == 404
    assert exc_info.value.payload == {
        "error": {"code": "not_found", "message": "Function not found"}
    }
    assert "Function not found" in str(exc_info.value)


def test_client_module_does_not_define_admin_routes() -> None:
    assert not hasattr(client_module, "StabilizerAdminClient")
    assert not hasattr(client_module.StabilizerClient, "evaluate_variance")
    assert not hasattr(client_module.StabilizerClient, "evaluate_ground_truth")

    client_source = Path(client_module.__file__).read_text(encoding="utf-8")
    assert "/v1/admin/" not in client_source
    assert "/v1/evaluate/" not in client_source

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


def test_wait_for_job_polls_until_completed_without_sleeping_after_terminal_status() -> None:
    transport = FakeTransport(
        [
            ResponseEnvelope(status_code=200, data={"job_id": "job_123", "status": "queued", "progress": 0}),
            ResponseEnvelope(status_code=200, data={"job_id": "job_123", "status": "running", "progress": 60}),
            ResponseEnvelope(
                status_code=200,
                data={"job_id": "job_123", "status": "completed", "progress": 100, "result": {"ok": True}},
            ),
        ]
    )
    sleeps: list[float] = []
    client = StabilizerClient(api_key="sk_test", transport=transport, sleeper=sleeps.append)

    job = client.wait_for_job("job_123", poll_interval=2.5, timeout=20.0)

    assert job["status"] == "completed"
    assert job["result"] == {"ok": True}
    assert [request.url for request in transport.requests] == [
        "https://stabilizerapi.documentinsight.ai/api/v1/jobs/job_123",
        "https://stabilizerapi.documentinsight.ai/api/v1/jobs/job_123",
        "https://stabilizerapi.documentinsight.ai/api/v1/jobs/job_123",
    ]
    assert sleeps == [2.5, 2.5]


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

    client_source = Path(client_module.__file__).read_text(encoding="utf-8")
    assert "/v1/admin/" not in client_source

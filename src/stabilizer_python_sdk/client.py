from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_BASE_URL = "https://stabilizerapi.documentinsight.ai/api"
DEFAULT_TIMEOUT = 30.0
PUBLIC_PATHS = frozenset({"/v1/health", "/v1/supported-models"})
@dataclass(frozen=True)
class ResponseEnvelope:
    status_code: int
    data: Any


class ApiError(RuntimeError):
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self.payload = payload
        message = self._extract_message(payload)
        super().__init__(f"Stabilizer API error {status_code}: {message}")

    @staticmethod
    def _extract_message(payload: Any) -> str:
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message:
                    return message
        return "request failed"


class Transport(Protocol):
    def send(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        json_body: object | None,
        timeout: float,
    ) -> ResponseEnvelope: ...


class UrlLibTransport:
    def send(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        json_body: object | None,
        timeout: float,
    ) -> ResponseEnvelope:
        data = None
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")

        request = Request(url=url, data=data, method=method)
        for key, value in headers.items():
            request.add_header(key, value)

        try:
            with urlopen(request, timeout=timeout) as response:
                return ResponseEnvelope(
                    status_code=response.status,
                    data=_decode_response_body(response.read()),
                )
        except HTTPError as exc:
            return ResponseEnvelope(
                status_code=exc.code,
                data=_decode_response_body(exc.read()),
            )
        except URLError as exc:
            raise RuntimeError(f"Unable to reach Stabilizer API: {exc.reason}") from exc


def _decode_response_body(body: bytes) -> Any:
    if not body:
        return None

    text = body.decode("utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _normalize_path(path: str) -> str:
    normalized = urlsplit(path).path.rstrip("/")
    return normalized or "/"


def _coerce_json_body(payload: object | None) -> object | None:
    if payload is None:
        return None
    as_payload = getattr(payload, "as_payload", None)
    if callable(as_payload):
        return as_payload()
    return payload


class _BaseClient:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        transport: Transport | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._transport = transport or UrlLibTransport()
        self.timeout = float(timeout)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: object | None = None,
        query: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        if query:
            query_items = {
                key: value
                for key, value in query.items()
                if value is not None and value != ""
            }
            if query_items:
                url = f"{url}?{urlencode(query_items, doseq=True)}"

        has_body = json_body is not None
        response = self._transport.send(
            method=method,
            url=url,
            headers=self._build_headers(path, has_body),
            json_body=json_body,
            timeout=self.timeout,
        )
        if not 200 <= response.status_code < 300:
            raise ApiError(response.status_code, response.data)
        return response.data

    def _build_headers(self, path: str, has_body: bool) -> dict[str, str]:
        headers = self._auth_headers(path)
        if has_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _auth_headers(self, path: str) -> dict[str, str]:
        return {}


class StabilizerClient(_BaseClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        transport: Transport | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(base_url=base_url, transport=transport, timeout=timeout)
        self.api_key = api_key

    def _auth_headers(self, path: str) -> dict[str, str]:
        if self.api_key and _normalize_path(path) not in PUBLIC_PATHS:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/v1/health")

    def supported_models(self) -> dict[str, Any]:
        return self._request("GET", "/v1/supported-models")

    def get_org(self) -> dict[str, Any]:
        return self._request("GET", "/v1/org")

    def update_org(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("PATCH", "/v1/org", json_body=payload)

    def list_api_keys(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v1/api-keys")

    def create_api_key(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/api-keys", json_body=payload)

    def revoke_api_key(self, key_id: str) -> None:
        self._request("DELETE", f"/v1/api-keys/{key_id}")
        return None

    def list_llm_configs(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v1/llm-configs")

    def create_llm_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/llm-configs", json_body=_coerce_json_body(payload))

    def update_llm_config(self, config_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("PATCH", f"/v1/llm-configs/{config_id}", json_body=payload)

    def delete_llm_config(self, config_id: str) -> None:
        self._request("DELETE", f"/v1/llm-configs/{config_id}")
        return None

    def list_functions(self, *, name: str | None = None, tag: str | None = None) -> list[dict[str, Any]]:
        return self._request("GET", "/v1/functions", query={"name": name, "tag": tag})

    def get_function(self, function_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/functions/{function_id}")

    def update_function(self, function_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("PATCH", f"/v1/functions/{function_id}", json_body=payload)

    def delete_function(self, function_id: str) -> None:
        self._request("DELETE", f"/v1/functions/{function_id}")
        return None

    def compile_function(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/functions", json_body=_coerce_json_body(payload))

    def optimize_prompt(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/prompt-optimizations", json_body=_coerce_json_body(payload))

    def extract(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/extract", json_body=_coerce_json_body(payload))

    def list_extractions(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v1/extractions")

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/jobs/{job_id}")

    def get_usage(self, *, from_: str | None = None, to: str | None = None) -> dict[str, Any]:
        return self._request("GET", "/v1/usage", query={"from": from_, "to": to})

    def evaluate_variance(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/evaluate/variance", json_body=payload)

    def evaluate_ground_truth(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/evaluate/gt", json_body=payload)

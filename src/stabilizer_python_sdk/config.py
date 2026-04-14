from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from stabilizer_python_sdk.workflow_runtime import (
    WorkflowConsole,
    default_now_provider,
    get_step_result,
    is_step_complete,
    load_state,
    save_state,
    update_step_state,
)


class SupportsConfigClient(Protocol):
    def create_llm_config(self, payload: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class LLMConfigRequest:
    name: str
    provider: str
    default_model: str
    api_key: str | None = None
    is_default: bool = False
    byok: bool = False

    def as_payload(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "provider": self.provider,
            "default_model": self.default_model,
            "is_default": self.is_default,
            "byok": self.byok,
        }
        if self.api_key not in (None, ""):
            payload["api_key"] = self.api_key
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> LLMConfigRequest:
        return cls(
            name=str(payload["name"]),
            provider=str(payload["provider"]),
            api_key=str(payload["api_key"]) if payload.get("api_key") is not None else None,
            default_model=str(payload["default_model"]),
            is_default=bool(payload.get("is_default", False)),
            byok=bool(payload.get("byok", False)),
        )


def create_llm_config(
    client: SupportsConfigClient,
    *,
    name: str,
    provider: str,
    api_key: str | None = None,
    default_model: str,
    is_default: bool = False,
    byok: bool = False,
) -> dict[str, Any]:
    request = LLMConfigRequest(
        name=name,
        provider=provider,
        api_key=api_key,
        default_model=default_model,
        is_default=is_default,
        byok=byok,
    )
    return client.create_llm_config(request.as_payload())


def run_config_step(
    client: SupportsConfigClient,
    *,
    request: LLMConfigRequest | dict[str, Any] | None = None,
    state_file: str | Path | None = None,
    temp_db_dir: str | Path = "temp_db",
    console: WorkflowConsole | None = None,
    now_provider=default_now_provider,
) -> dict[str, Any]:
    workflow_console = console or WorkflowConsole()
    path, state = load_state(
        state_file=state_file,
        temp_db_dir=temp_db_dir,
        now_provider=now_provider,
    )

    if is_step_complete(state, "config"):
        workflow_console.skip("config", f"Skipping existing result from {path.name}.")
        saved_result = get_step_result(state, "config")
        if saved_result is None:
            raise ValueError("Saved config step is missing its result.")
        return saved_result

    resolved_request = _resolve_config_request(request)
    workflow_console.section("config", "Creating LLM config.")
    result = client.create_llm_config(resolved_request.as_payload())
    update_step_state(
        state,
        step_name="config",
        request=resolved_request.as_payload(),
        result=result,
        extra_fields={"config_id": result.get("config_id")},
    )
    save_state(path, state, now_provider=now_provider)
    workflow_console.success("config", f"Saved config result to {path}.")
    return result


def _resolve_config_request(
    request: LLMConfigRequest | dict[str, Any] | None,
) -> LLMConfigRequest:
    if isinstance(request, LLMConfigRequest):
        return request
    if isinstance(request, dict):
        return LLMConfigRequest.from_payload(request)
    raise ValueError("Config step requires an LLMConfigRequest or payload dict.")

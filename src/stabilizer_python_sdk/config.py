from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class SupportsConfigClient(Protocol):
    def create_llm_config(self, payload: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class LLMConfigRequest:
    name: str
    provider: str
    api_key: str
    default_model: str
    is_default: bool = False

    def as_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "api_key": self.api_key,
            "default_model": self.default_model,
            "is_default": self.is_default,
        }


def create_llm_config(
    client: SupportsConfigClient,
    *,
    name: str,
    provider: str,
    api_key: str,
    default_model: str,
    is_default: bool = False,
) -> dict[str, Any]:
    request = LLMConfigRequest(
        name=name,
        provider=provider,
        api_key=api_key,
        default_model=default_model,
        is_default=is_default,
    )
    return client.create_llm_config(request.as_payload())

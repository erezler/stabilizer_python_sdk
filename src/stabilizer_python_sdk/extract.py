from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


class SupportsExtractClient(Protocol):
    def extract(self, payload: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ExtractOptions:
    num_results: int

    def as_payload(self) -> dict[str, Any]:
        return {"num_results": self.num_results}


@dataclass(frozen=True)
class ExtractRequest:
    function_id: str
    source_text: str
    options: ExtractOptions | Mapping[str, Any] | None = None

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "function_id": self.function_id,
            "source_text": self.source_text,
        }
        if self.options is not None:
            payload["options"] = _serialize_extract_options(self.options)
        return payload


def extract(
    client: SupportsExtractClient,
    *,
    function_id: str,
    source_text: str,
    options: ExtractOptions | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    request = ExtractRequest(
        function_id=function_id,
        source_text=source_text,
        options=options,
    )
    return client.extract(request.as_payload())


def _serialize_extract_options(options: ExtractOptions | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(options, ExtractOptions):
        return options.as_payload()
    return dict(options)

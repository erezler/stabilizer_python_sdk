from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class SupportsOptimizeClient(Protocol):
    def optimize_prompt(self, payload: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class TrainingExample:
    source_text: str
    extracted_json: dict[str, Any]

    def as_payload(self) -> dict[str, Any]:
        return {
            "source_text": self.source_text,
            "extracted_json": self.extracted_json,
        }


@dataclass(frozen=True)
class OptimizeRequest:
    prompt: str
    json_structure: dict[str, Any]
    training_data: Sequence[TrainingExample | Mapping[str, Any]]

    def as_payload(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "json_structure": self.json_structure,
            "training_data": [_serialize_training_example(example) for example in self.training_data],
        }


def optimize_prompt(
    client: SupportsOptimizeClient,
    *,
    prompt: str,
    json_structure: dict[str, Any],
    training_data: Sequence[TrainingExample | Mapping[str, Any]],
) -> dict[str, Any]:
    request = OptimizeRequest(
        prompt=prompt,
        json_structure=json_structure,
        training_data=training_data,
    )
    return client.optimize_prompt(request.as_payload())


def _serialize_training_example(example: TrainingExample | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(example, TrainingExample):
        return example.as_payload()
    return dict(example)

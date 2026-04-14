from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from stabilizer_python_sdk.optimize import TrainingExample


class SupportsCompileClient(Protocol):
    def compile_function(self, payload: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class CompileOptions:
    num_prompt_variations: int

    def as_payload(self) -> dict[str, Any]:
        return {"num_prompt_variations": self.num_prompt_variations}


@dataclass(frozen=True)
class CompileRequest:
    name: str
    prompt: str
    json_structure: dict[str, Any]
    description: str | None = None
    tags: Sequence[str] = ()
    training_data: Sequence[TrainingExample | Mapping[str, Any]] = ()
    grounding_methods: Sequence[str] = ()
    compile_options: CompileOptions | Mapping[str, Any] | None = None

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "prompt": self.prompt,
            "json_structure": self.json_structure,
        }
        if self.description is not None:
            payload["description"] = self.description
        if self.tags:
            payload["tags"] = list(self.tags)
        if self.training_data:
            payload["training_data"] = [_serialize_training_example(example) for example in self.training_data]
        if self.grounding_methods:
            payload["grounding_methods"] = list(self.grounding_methods)
        if self.compile_options is not None:
            payload["compile_options"] = _serialize_compile_options(self.compile_options)
        return payload


def compile_function(
    client: SupportsCompileClient,
    *,
    name: str,
    prompt: str,
    json_structure: dict[str, Any],
    description: str | None = None,
    tags: Sequence[str] = (),
    training_data: Sequence[TrainingExample | Mapping[str, Any]] = (),
    grounding_methods: Sequence[str] = (),
    compile_options: CompileOptions | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    request = CompileRequest(
        name=name,
        description=description,
        tags=tags,
        prompt=prompt,
        json_structure=json_structure,
        training_data=training_data,
        grounding_methods=grounding_methods,
        compile_options=compile_options,
    )
    return client.compile_function(request.as_payload())


def _serialize_training_example(example: TrainingExample | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(example, TrainingExample):
        return example.as_payload()
    return dict(example)


def _serialize_compile_options(options: CompileOptions | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(options, CompileOptions):
        return options.as_payload()
    return dict(options)

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from stabilizer_python_sdk.optimize import TrainingExample
from stabilizer_python_sdk.workflow_runtime import (
    WorkflowConsole,
    default_now_provider,
    get_saved_request,
    get_step_result,
    is_step_complete,
    load_state,
    poll_job,
    save_state,
    update_step_state,
)


class SupportsCompileClient(Protocol):
    def compile_function(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    def get_job(self, job_id: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class CompileOptions:
    llm_config_id: str | None = None
    compile_model: str | None = None
    use_agents_network: bool | None = None
    force_agents_network_sequential: bool | None = None
    min_field_pass_rate: float | None = None
    min_overall_pass_rate: float | None = None
    num_prompt_variations: int | None = None
    optimized_prompts: Sequence[str] = ()
    compile_mode: str | None = None

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.llm_config_id not in (None, ""):
            payload["llm_config_id"] = self.llm_config_id
        if self.compile_model not in (None, ""):
            payload["compile_model"] = self.compile_model
        if self.use_agents_network is not None:
            payload["use_agents_network"] = self.use_agents_network
        if self.force_agents_network_sequential is not None:
            payload["force_agents_network_sequential"] = self.force_agents_network_sequential
        if self.min_field_pass_rate is not None:
            payload["min_field_pass_rate"] = self.min_field_pass_rate
        if self.min_overall_pass_rate is not None:
            payload["min_overall_pass_rate"] = self.min_overall_pass_rate
        if self.num_prompt_variations is not None:
            payload["num_prompt_variations"] = self.num_prompt_variations
        if self.optimized_prompts:
            payload["optimized_prompts"] = list(self.optimized_prompts)
        if self.compile_mode not in (None, ""):
            payload["compile_mode"] = self.compile_mode
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CompileOptions:
        return cls(
            llm_config_id=str(payload["llm_config_id"]) if payload.get("llm_config_id") is not None else None,
            compile_model=str(payload["compile_model"]) if payload.get("compile_model") is not None else None,
            use_agents_network=(
                bool(payload["use_agents_network"])
                if payload.get("use_agents_network") is not None
                else None
            ),
            force_agents_network_sequential=(
                bool(payload["force_agents_network_sequential"])
                if payload.get("force_agents_network_sequential") is not None
                else None
            ),
            min_field_pass_rate=(
                float(payload["min_field_pass_rate"])
                if payload.get("min_field_pass_rate") is not None
                else None
            ),
            min_overall_pass_rate=(
                float(payload["min_overall_pass_rate"])
                if payload.get("min_overall_pass_rate") is not None
                else None
            ),
            num_prompt_variations=(
                int(payload["num_prompt_variations"])
                if payload.get("num_prompt_variations") is not None
                else None
            ),
            optimized_prompts=[
                str(prompt)
                for prompt in payload.get("optimized_prompts", [])
            ],
            compile_mode=str(payload["compile_mode"]) if payload.get("compile_mode") is not None else None,
        )


@dataclass(frozen=True)
class CompileRequest:
    prompt: str
    json_structure: dict[str, Any]
    name: str | None = None
    description: str | None = None
    tags: Sequence[str] = ()
    training_data: Sequence[TrainingExample | Mapping[str, Any]] = ()
    grounding_methods: Sequence[str] = ()
    compile_options: CompileOptions | Mapping[str, Any] | None = None

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": self.prompt,
            "json_structure": self.json_structure,
        }
        if self.name not in (None, ""):
            payload["name"] = self.name
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

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> CompileRequest:
        training_data = payload.get("training_data", [])
        if not isinstance(training_data, Sequence):
            raise ValueError("Compile request training_data must be a sequence.")
        compile_options = payload.get("compile_options")
        return cls(
            name=str(payload["name"]) if payload.get("name") is not None else None,
            description=str(payload["description"]) if payload.get("description") is not None else None,
            tags=list(payload.get("tags", [])),
            prompt=str(payload["prompt"]),
            json_structure=dict(payload["json_structure"]),
            training_data=[
                TrainingExample.from_payload(example) if isinstance(example, Mapping) else example
                for example in training_data
            ],
            grounding_methods=list(payload.get("grounding_methods", [])),
            compile_options=(
                CompileOptions.from_payload(compile_options)
                if isinstance(compile_options, Mapping)
                else compile_options
            ),
        )


def compile_function(
    client: SupportsCompileClient,
    *,
    prompt: str,
    json_structure: dict[str, Any],
    name: str | None = None,
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


def run_compile_step(
    client: SupportsCompileClient,
    *,
    request: CompileRequest | dict[str, Any] | None = None,
    state_file: str | Path | None = None,
    temp_db_dir: str | Path = "temp_db",
    console: WorkflowConsole | None = None,
    poll_interval: float = 2.0,
    timeout: float = 600.0,
    now_provider=default_now_provider,
    sleeper=None,
    monotonic=None,
) -> dict[str, Any]:
    workflow_console = console or WorkflowConsole()
    path, state = load_state(
        state_file=state_file,
        temp_db_dir=temp_db_dir,
        now_provider=now_provider,
    )

    if is_step_complete(state, "compile"):
        workflow_console.skip("compile", f"Skipping existing result from {path.name}.")
        saved_result = get_step_result(state, "compile")
        if saved_result is None:
            raise ValueError("Saved compile step is missing its result.")
        return saved_result

    resolved_request = _resolve_compile_request(request, state)
    workflow_console.section("compile", "Submitting compile job.")
    submission = client.compile_function(resolved_request.as_payload())
    final_result = poll_job(
        client,
        job_id=str(submission["job_id"]),
        step_name="compile",
        console=workflow_console,
        poll_interval=poll_interval,
        timeout=timeout,
        sleeper=sleeper or time.sleep,
        monotonic=monotonic or time.monotonic,
    )
    function_id = _extract_function_id(final_result)
    update_step_state(
        state,
        step_name="compile",
        request=resolved_request.as_payload(),
        submission=submission,
        result=final_result,
        extra_fields={
            "job_id": submission.get("job_id"),
            "function_id": function_id,
        },
    )
    save_state(path, state, now_provider=now_provider)
    workflow_console.success("compile", f"Saved compile result to {path}.")
    return final_result


def _serialize_training_example(example: TrainingExample | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(example, TrainingExample):
        return example.as_payload()
    return dict(example)


def _serialize_compile_options(options: CompileOptions | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(options, CompileOptions):
        return options.as_payload()
    return dict(options)


def _resolve_compile_request(
    request: CompileRequest | dict[str, Any] | None,
    state: dict[str, Any],
) -> CompileRequest:
    if isinstance(request, CompileRequest):
        return request
    if isinstance(request, dict):
        return CompileRequest.from_payload(request)

    saved_request = get_saved_request(state, "compile")
    if saved_request is not None:
        return CompileRequest.from_payload(saved_request)

    raise ValueError("Compile step requires a CompileRequest or payload dict.")


def _extract_function_id(result: Mapping[str, Any]) -> str | None:
    nested_result = result.get("result")
    if isinstance(nested_result, Mapping):
        function_id = nested_result.get("function_id")
        if function_id is not None:
            return str(function_id)
    return None

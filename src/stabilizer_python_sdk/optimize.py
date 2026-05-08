from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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


class SupportsOptimizeClient(Protocol):
    def optimize_prompt(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    def get_job(self, job_id: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class TrainingExample:
    source_text: str
    extracted_json: dict[str, Any]

    def as_payload(self) -> dict[str, Any]:
        return {
            "source_text": self.source_text,
            "extracted_json": self.extracted_json,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TrainingExample:
        return cls(
            source_text=str(payload["source_text"]),
            extracted_json=dict(payload["extracted_json"]),
        )


@dataclass(frozen=True)
class PromptOptimizationOptions:
    llm_config_id: str | None = None
    optimization_model: str | None = None
    num_optimized_prompts: int | None = None
    max_training_examples: int | None = None
    max_bootstrapped_demos: int | None = None
    max_labeled_demos: int | None = None
    max_rounds: int | None = None
    num_candidate_programs: int | None = None
    variation_attempt_multiplier: int | None = None

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.llm_config_id not in (None, ""):
            payload["llm_config_id"] = self.llm_config_id
        if self.optimization_model not in (None, ""):
            payload["optimization_model"] = self.optimization_model
        if self.num_optimized_prompts is not None:
            payload["num_optimized_prompts"] = self.num_optimized_prompts
        if self.max_training_examples is not None:
            payload["max_training_examples"] = self.max_training_examples
        if self.max_bootstrapped_demos is not None:
            payload["max_bootstrapped_demos"] = self.max_bootstrapped_demos
        if self.max_labeled_demos is not None:
            payload["max_labeled_demos"] = self.max_labeled_demos
        if self.max_rounds is not None:
            payload["max_rounds"] = self.max_rounds
        if self.num_candidate_programs is not None:
            payload["num_candidate_programs"] = self.num_candidate_programs
        if self.variation_attempt_multiplier is not None:
            payload["variation_attempt_multiplier"] = self.variation_attempt_multiplier
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PromptOptimizationOptions:
        return cls(
            llm_config_id=str(payload["llm_config_id"]) if payload.get("llm_config_id") is not None else None,
            optimization_model=(
                str(payload["optimization_model"])
                if payload.get("optimization_model") is not None
                else None
            ),
            num_optimized_prompts=(
                int(payload["num_optimized_prompts"])
                if payload.get("num_optimized_prompts") is not None
                else None
            ),
            max_training_examples=(
                int(payload["max_training_examples"])
                if payload.get("max_training_examples") is not None
                else None
            ),
            max_bootstrapped_demos=(
                int(payload["max_bootstrapped_demos"])
                if payload.get("max_bootstrapped_demos") is not None
                else None
            ),
            max_labeled_demos=(
                int(payload["max_labeled_demos"])
                if payload.get("max_labeled_demos") is not None
                else None
            ),
            max_rounds=int(payload["max_rounds"]) if payload.get("max_rounds") is not None else None,
            num_candidate_programs=(
                int(payload["num_candidate_programs"])
                if payload.get("num_candidate_programs") is not None
                else None
            ),
            variation_attempt_multiplier=(
                int(payload["variation_attempt_multiplier"])
                if payload.get("variation_attempt_multiplier") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class OptimizeRequest:
    prompt: str
    json_structure: dict[str, Any]
    training_data: Sequence[TrainingExample | Mapping[str, Any]]
    optimization_options: PromptOptimizationOptions | Mapping[str, Any] | None = None

    def as_payload(self) -> dict[str, Any]:
        payload = {
            "prompt": self.prompt,
            "json_structure": self.json_structure,
            "training_data": [_serialize_training_example(example) for example in self.training_data],
        }
        if self.optimization_options is not None:
            payload["optimization_options"] = _serialize_optimization_options(self.optimization_options)
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> OptimizeRequest:
        training_data = payload.get("training_data", [])
        if not isinstance(training_data, Sequence):
            raise ValueError("Optimize request training_data must be a sequence.")
        optimization_options = payload.get("optimization_options")
        return cls(
            prompt=str(payload["prompt"]),
            json_structure=dict(payload["json_structure"]),
            training_data=[
                TrainingExample.from_payload(example) if isinstance(example, Mapping) else example
                for example in training_data
            ],
            optimization_options=(
                PromptOptimizationOptions.from_payload(optimization_options)
                if isinstance(optimization_options, Mapping)
                else optimization_options
            ),
        )


def optimize_prompt(
    client: SupportsOptimizeClient,
    *,
    prompt: str,
    json_structure: dict[str, Any],
    training_data: Sequence[TrainingExample | Mapping[str, Any]],
    optimization_options: PromptOptimizationOptions | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    request = OptimizeRequest(
        prompt=prompt,
        json_structure=json_structure,
        training_data=training_data,
        optimization_options=optimization_options,
    )
    return client.optimize_prompt(request.as_payload())


def run_optimize_step(
    client: SupportsOptimizeClient,
    *,
    request: OptimizeRequest | dict[str, Any] | None = None,
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

    if is_step_complete(state, "optimize"):
        workflow_console.skip("optimize", f"Skipping existing result from {path.name}.")
        saved_result = get_step_result(state, "optimize")
        if saved_result is None:
            raise ValueError("Saved optimize step is missing its result.")
        return saved_result

    resolved_request = _resolve_optimize_request(request, state)
    workflow_console.section("optimize", "Submitting prompt optimization job.")
    submission = client.optimize_prompt(resolved_request.as_payload())
    final_result = poll_job(
        client,
        job_id=str(submission["job_id"]),
        step_name="optimize",
        console=workflow_console,
        poll_interval=poll_interval,
        timeout=timeout,
        sleeper=sleeper or time.sleep,
        monotonic=monotonic or time.monotonic,
    )
    update_step_state(
        state,
        step_name="optimize",
        request=resolved_request.as_payload(),
        submission=submission,
        result=final_result,
        extra_fields={"job_id": submission.get("job_id")},
    )
    save_state(path, state, now_provider=now_provider)
    workflow_console.success("optimize", f"Saved optimization result to {path}.")
    return final_result


def _serialize_training_example(example: TrainingExample | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(example, TrainingExample):
        return example.as_payload()
    return dict(example)


def _serialize_optimization_options(
    options: PromptOptimizationOptions | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(options, PromptOptimizationOptions):
        return options.as_payload()
    return dict(options)


def _resolve_optimize_request(
    request: OptimizeRequest | dict[str, Any] | None,
    state: dict[str, Any],
) -> OptimizeRequest:
    if isinstance(request, OptimizeRequest):
        return request
    if isinstance(request, dict):
        return OptimizeRequest.from_payload(request)

    saved_request = get_saved_request(state, "optimize")
    if saved_request is not None:
        return OptimizeRequest.from_payload(saved_request)

    raise ValueError("Optimize step requires an OptimizeRequest or payload dict.")

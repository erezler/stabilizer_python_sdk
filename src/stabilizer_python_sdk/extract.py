from __future__ import annotations

import time
from collections.abc import Mapping
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


class SupportsExtractClient(Protocol):
    def extract(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    def get_job(self, job_id: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ExtractOptions:
    num_results: int

    def as_payload(self) -> dict[str, Any]:
        return {"num_results": self.num_results}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ExtractOptions:
        return cls(num_results=int(payload["num_results"]))


@dataclass(frozen=True)
class ExtractRequest:
    function_id: str | None
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

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ExtractRequest:
        options = payload.get("options")
        return cls(
            function_id=str(payload["function_id"]) if payload.get("function_id") is not None else None,
            source_text=str(payload["source_text"]),
            options=ExtractOptions.from_payload(options) if isinstance(options, Mapping) else options,
        )


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


def run_extract_step(
    client: SupportsExtractClient,
    *,
    request: ExtractRequest | dict[str, Any] | None = None,
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

    if is_step_complete(state, "extract"):
        workflow_console.skip("extract", f"Skipping existing result from {path.name}.")
        saved_result = get_step_result(state, "extract")
        if saved_result is None:
            raise ValueError("Saved extract step is missing its result.")
        return saved_result

    resolved_request = _resolve_extract_request(request, state)
    workflow_console.section("extract", "Submitting extraction job.")
    submission = client.extract(resolved_request.as_payload())
    final_result = poll_job(
        client,
        job_id=str(submission["job_id"]),
        step_name="extract",
        console=workflow_console,
        poll_interval=poll_interval,
        timeout=timeout,
        sleeper=sleeper or time.sleep,
        monotonic=monotonic or time.monotonic,
    )
    update_step_state(
        state,
        step_name="extract",
        request=resolved_request.as_payload(),
        submission=submission,
        result=final_result,
        extra_fields={"job_id": submission.get("job_id")},
    )
    save_state(path, state, now_provider=now_provider)
    workflow_console.success("extract", f"Saved extract result to {path}.")
    return final_result


def _serialize_extract_options(options: ExtractOptions | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(options, ExtractOptions):
        return options.as_payload()
    return dict(options)


def _resolve_extract_request(
    request: ExtractRequest | dict[str, Any] | None,
    state: dict[str, Any],
) -> ExtractRequest:
    candidate: ExtractRequest
    if isinstance(request, ExtractRequest):
        candidate = request
    elif isinstance(request, dict):
        candidate = ExtractRequest.from_payload(request)
    else:
        saved_request = get_saved_request(state, "extract")
        if saved_request is None:
            raise ValueError("Extract step requires an ExtractRequest or payload dict.")
        candidate = ExtractRequest.from_payload(saved_request)

    if candidate.function_id in (None, "", "fn_replace_me"):
        compile_step = state.get("steps", {}).get("compile", {})
        function_id = None
        if isinstance(compile_step, dict):
            raw_function_id = compile_step.get("function_id")
            if raw_function_id is not None:
                function_id = str(raw_function_id)
        if function_id is None:
            raise ValueError("Extract step requires a function_id or a saved compile result.")
        candidate = ExtractRequest(
            function_id=function_id,
            source_text=candidate.source_text,
            options=candidate.options,
        )
    return candidate

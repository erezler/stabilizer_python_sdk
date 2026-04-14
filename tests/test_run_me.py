from __future__ import annotations

import io
import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from stabilizer_python_sdk.compile import CompileRequest, run_compile_step
from stabilizer_python_sdk.config import LLMConfigRequest, run_config_step
from stabilizer_python_sdk.extract import ExtractOptions, ExtractRequest
from stabilizer_python_sdk.optimize import OptimizeRequest, TrainingExample
from stabilizer_python_sdk.run_me import RunMeSettings, run_all
from stabilizer_python_sdk.workflow_runtime import WorkflowConsole


class FakeWorkflowRunClient:
    def __init__(self) -> None:
        self.config_calls: list[dict[str, object]] = []
        self.optimize_calls: list[dict[str, object]] = []
        self.compile_calls: list[dict[str, object]] = []
        self.extract_calls: list[dict[str, object]] = []
        self.jobs: dict[str, Iterator[dict[str, object]]] = {
            "job_opt": iter(
                [
                    {"job_id": "job_opt", "status": "queued", "progress": 15},
                    {
                        "job_id": "job_opt",
                        "status": "completed",
                        "progress": 100,
                        "result": {"optimized_prompt": "Optimized"},
                    },
                ]
            ),
            "job_compile": iter(
                [
                    {"job_id": "job_compile", "status": "running", "progress": 40},
                    {
                        "job_id": "job_compile",
                        "status": "completed",
                        "progress": 100,
                        "result": {"function_id": "fn_123"},
                    },
                ]
            ),
            "job_extract": iter(
                [
                    {"job_id": "job_extract", "status": "running", "progress": 75},
                    {
                        "job_id": "job_extract",
                        "status": "completed",
                        "progress": 100,
                        "result": {"event_title": "Harbor Lights Food Fair"},
                    },
                ]
            ),
        }

    def create_llm_config(self, payload: dict[str, object]) -> dict[str, object]:
        self.config_calls.append(payload)
        return {"config_id": "cfg_123", "status": "created"}

    def optimize_prompt(self, payload: dict[str, object]) -> dict[str, object]:
        self.optimize_calls.append(payload)
        return {"job_id": "job_opt", "status": "queued"}

    def compile_function(self, payload: dict[str, object]) -> dict[str, object]:
        self.compile_calls.append(payload)
        return {"job_id": "job_compile", "status": "queued"}

    def extract(self, payload: dict[str, object]) -> dict[str, object]:
        self.extract_calls.append(payload)
        return {"job_id": "job_extract", "status": "queued"}

    def get_job(self, job_id: str) -> dict[str, object]:
        return next(self.jobs[job_id])


def _fixed_now() -> datetime:
    return datetime(2026, 4, 14, 8, 30, 45)


def test_config_step_creates_state_file_and_skips_when_result_exists(tmp_path: Path) -> None:
    client = FakeWorkflowRunClient()
    temp_db_dir = tmp_path / "temp_db"
    output = io.StringIO()
    console = WorkflowConsole(stream=output)

    created = run_config_step(
        client,
        request=LLMConfigRequest(
            name="Primary config",
            provider="openai",
            api_key="provider-key",
            default_model="openai/gpt-5.4-mini",
            is_default=True,
        ),
        temp_db_dir=temp_db_dir,
        console=console,
        now_provider=_fixed_now,
    )
    skipped = run_config_step(
        client,
        temp_db_dir=temp_db_dir,
        console=console,
        now_provider=_fixed_now,
    )

    state_file = temp_db_dir / "2026-04-14-08-30-45.json"
    state = json.loads(state_file.read_text(encoding="utf-8"))

    assert created == {"config_id": "cfg_123", "status": "created"}
    assert skipped == {"config_id": "cfg_123", "status": "created"}
    assert client.config_calls == [
        {
            "name": "Primary config",
            "provider": "openai",
            "api_key": "provider-key",
            "default_model": "openai/gpt-5.4-mini",
            "is_default": True,
        }
    ]
    assert state["steps"]["config"]["config_id"] == "cfg_123"
    assert state["steps"]["config"]["request"]["name"] == "Primary config"
    assert "\x1b[" in output.getvalue()
    assert "Skipping" in output.getvalue()


def test_run_all_loads_requests_updates_same_state_file_and_uses_compiled_function_id(
    tmp_path: Path,
) -> None:
    client = FakeWorkflowRunClient()
    output = io.StringIO()
    console = WorkflowConsole(stream=output)
    compile_payload_path = tmp_path / "compile.json"
    extract_payload_path = tmp_path / "extract.json"
    temp_db_dir = tmp_path / "temp_db"

    compile_payload_path.write_text(
        json.dumps(
            {
                "name": "Event details extractor",
                "description": "Extracts event details from text",
                "tags": ["events", "walkthrough"],
                "prompt": "Extract event details",
                "json_structure": {"event_title": "string"},
                "training_data": [
                    {
                        "source_text": "The event is tomorrow.",
                        "extracted_json": {"event_title": "Tomorrow Event"},
                    }
                ],
                "grounding_methods": ["hard_grounding", "constraints_validation"],
                "compile_options": {"num_prompt_variations": 3},
            }
        ),
        encoding="utf-8",
    )
    extract_payload_path.write_text(
        json.dumps(
            {
                "function_id": "fn_replace_me",
                "source_text": "The Harbor Lights Food Fair returns tomorrow.",
                "options": {"num_results": 3},
            }
        ),
        encoding="utf-8",
    )

    state = run_all(
        settings=RunMeSettings(
            api_key="sk_test",
            temp_db_dir=temp_db_dir,
            config_request=LLMConfigRequest(
                name="Primary config",
                provider="openai",
                api_key="provider-key",
                default_model="openai/gpt-5.4-mini",
                is_default=True,
            ),
            optimize_request=OptimizeRequest(
                prompt="Extract event details",
                json_structure={"event_title": "string"},
                training_data=[
                    TrainingExample(
                        source_text="The event is tomorrow.",
                        extracted_json={"event_title": "Tomorrow Event"},
                    )
                ],
            ),
            compile_payload_file=compile_payload_path,
            extract_payload_file=extract_payload_path,
            poll_interval=0.0,
            poll_timeout=30.0,
        ),
        client=client,
        console=console,
        now_provider=_fixed_now,
        sleeper=lambda _seconds: None,
    )

    state_file = temp_db_dir / "2026-04-14-08-30-45.json"
    saved_state = json.loads(state_file.read_text(encoding="utf-8"))

    assert state == saved_state
    assert client.optimize_calls == [
        {
            "prompt": "Extract event details",
            "json_structure": {"event_title": "string"},
            "training_data": [
                {
                    "source_text": "The event is tomorrow.",
                    "extracted_json": {"event_title": "Tomorrow Event"},
                }
            ],
        }
    ]
    assert client.compile_calls == [
        {
            "name": "Event details extractor",
            "description": "Extracts event details from text",
            "tags": ["events", "walkthrough"],
            "prompt": "Extract event details",
            "json_structure": {"event_title": "string"},
            "training_data": [
                {
                    "source_text": "The event is tomorrow.",
                    "extracted_json": {"event_title": "Tomorrow Event"},
                }
            ],
            "grounding_methods": ["hard_grounding", "constraints_validation"],
            "compile_options": {"num_prompt_variations": 3},
        }
    ]
    assert client.extract_calls == [
        {
            "function_id": "fn_123",
            "source_text": "The Harbor Lights Food Fair returns tomorrow.",
            "options": {"num_results": 3},
        }
    ]
    assert saved_state["steps"]["config"]["config_id"] == "cfg_123"
    assert saved_state["steps"]["optimize"]["job_id"] == "job_opt"
    assert saved_state["steps"]["compile"]["function_id"] == "fn_123"
    assert saved_state["steps"]["extract"]["result"]["result"] == {
        "event_title": "Harbor Lights Food Fair"
    }
    assert "poll_history" not in saved_state["steps"]["optimize"]
    assert "100%" in output.getvalue()
    assert "\x1b[" in output.getvalue()

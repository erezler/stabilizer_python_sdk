from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from stabilizer_python_sdk import ResponseEnvelope, StabilizerClient
from stabilizer_python_sdk.compile import CompileOptions, CompileRequest, compile_function
from stabilizer_python_sdk.config import LLMConfigRequest, create_llm_config
from stabilizer_python_sdk.extract import ExtractOptions, ExtractRequest, extract
from stabilizer_python_sdk.optimize import (
    OptimizeRequest,
    PromptOptimizationOptions,
    TrainingExample,
    optimize_prompt,
)


class FakeWorkflowClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def create_llm_config(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("create_llm_config", payload))
        return {"config_id": "cfg_123"}

    def optimize_prompt(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("optimize_prompt", payload))
        return {"job_id": "job_opt"}

    def compile_function(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("compile_function", payload))
        return {"job_id": "job_compile"}

    def extract(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("extract", payload))
        return {"job_id": "job_extract"}


@dataclass
class RecordedRequest:
    method: str
    url: str
    headers: dict[str, str]
    json_body: object | None
    timeout: float


class FakeTransport:
    def __init__(self, responses: list[ResponseEnvelope]) -> None:
        self._responses: Iterator[ResponseEnvelope] = iter(responses)
        self.requests: list[RecordedRequest] = []

    def send(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        json_body: object | None,
        timeout: float,
    ) -> ResponseEnvelope:
        self.requests.append(
            RecordedRequest(
                method=method,
                url=url,
                headers=headers,
                json_body=json_body,
                timeout=timeout,
            )
        )
        return next(self._responses)


def test_config_module_accepts_explicit_parameters() -> None:
    client = FakeWorkflowClient()

    response = create_llm_config(
        client,
        name="Primary config",
        provider="openai",
        api_key="provider-key",
        default_model="google/gemini-2.5-flash-lite",
        is_default=True,
        byok=True,
    )

    assert response == {"config_id": "cfg_123"}
    assert client.calls == [
        (
            "create_llm_config",
            {
                "name": "Primary config",
                "provider": "openai",
                "api_key": "provider-key",
                "default_model": "google/gemini-2.5-flash-lite",
                "is_default": True,
                "byok": True,
            },
        )
    ]


def test_config_module_accepts_minimal_api_payload() -> None:
    client = FakeWorkflowClient()

    response = create_llm_config(
        client,
        name="Primary config",
    )

    assert response == {"config_id": "cfg_123"}
    assert client.calls == [
        (
            "create_llm_config",
            {
                "name": "Primary config",
            },
        )
    ]


def test_optimize_module_accepts_explicit_parameters() -> None:
    client = FakeWorkflowClient()

    response = optimize_prompt(
        client,
        prompt="Extract event details",
        json_structure={"event_title": "string"},
        training_data=[
            TrainingExample(
                source_text="The event is tomorrow.",
                extracted_json={"event_title": "Tomorrow Event"},
            )
        ],
    )

    assert response == {"job_id": "job_opt"}
    assert client.calls == [
        (
            "optimize_prompt",
            {
                "prompt": "Extract event details",
                "json_structure": {"event_title": "string"},
                "training_data": [
                    {
                        "source_text": "The event is tomorrow.",
                        "extracted_json": {"event_title": "Tomorrow Event"},
                    }
                ],
            },
        )
    ]


def test_optimize_module_accepts_optimization_options() -> None:
    client = FakeWorkflowClient()

    response = optimize_prompt(
        client,
        prompt="Extract event details",
        json_structure={"event_title": "string"},
        training_data=[
            TrainingExample(
                source_text="The event is tomorrow.",
                extracted_json={"event_title": "Tomorrow Event"},
            )
        ],
        optimization_options=PromptOptimizationOptions(
            llm_config_id="cfg_123",
            optimization_model="openai/gpt-5.4",
            num_optimized_prompts=4,
            max_training_examples=10,
        ),
    )

    assert response == {"job_id": "job_opt"}
    assert client.calls == [
        (
            "optimize_prompt",
            {
                "prompt": "Extract event details",
                "json_structure": {"event_title": "string"},
                "training_data": [
                    {
                        "source_text": "The event is tomorrow.",
                        "extracted_json": {"event_title": "Tomorrow Event"},
                    }
                ],
                "optimization_options": {
                    "llm_config_id": "cfg_123",
                    "optimization_model": "openai/gpt-5.4",
                    "num_optimized_prompts": 4,
                    "max_training_examples": 10,
                },
            },
        )
    ]


def test_compile_module_accepts_explicit_parameters() -> None:
    client = FakeWorkflowClient()

    response = compile_function(
        client,
        name="Event details extractor",
        description="Extracts event details from text",
        tags=["events", "walkthrough"],
        prompt="Extract event details",
        json_structure={"event_title": "string"},
        training_data=[
            TrainingExample(
                source_text="Source",
                extracted_json={"event_title": "Value"},
            )
        ],
        grounding_methods=["hard_grounding", "constraints_validation"],
        compile_options=CompileOptions(num_prompt_variations=3),
    )

    assert response == {"job_id": "job_compile"}
    assert client.calls == [
        (
            "compile_function",
            {
                "name": "Event details extractor",
                "description": "Extracts event details from text",
                "tags": ["events", "walkthrough"],
                "prompt": "Extract event details",
                "json_structure": {"event_title": "string"},
                "training_data": [
                    {
                        "source_text": "Source",
                        "extracted_json": {"event_title": "Value"},
                    }
                ],
                "grounding_methods": ["hard_grounding", "constraints_validation"],
                "compile_options": {"num_prompt_variations": 3},
            },
        )
    ]


def test_compile_module_accepts_optional_name_and_extended_compile_options() -> None:
    client = FakeWorkflowClient()

    response = compile_function(
        client,
        prompt="Extract event details",
        json_structure={"event_title": "string"},
        compile_options=CompileOptions(
            llm_config_id="cfg_123",
            compile_strength="high",
            use_agents_network=True,
            force_agents_network_sequential=True,
            min_field_pass_rate=0.75,
            min_overall_pass_rate=0.8,
            optimized_prompts=["Prompt A", "Prompt B"],
            compile_mode="agents_network",
        ),
    )

    assert response == {"job_id": "job_compile"}
    assert client.calls == [
        (
            "compile_function",
            {
                "prompt": "Extract event details",
                "json_structure": {"event_title": "string"},
                "compile_options": {
                    "llm_config_id": "cfg_123",
                    "compile_strength": "high",
                    "use_agents_network": True,
                    "force_agents_network_sequential": True,
                    "min_field_pass_rate": 0.75,
                    "min_overall_pass_rate": 0.8,
                    "optimized_prompts": ["Prompt A", "Prompt B"],
                    "compile_mode": "agents_network",
                },
            },
        )
    ]


def test_extract_module_accepts_explicit_parameters() -> None:
    client = FakeWorkflowClient()

    response = extract(
        client,
        function_id="fn_123",
        source_text="hello",
        options=ExtractOptions(num_results=3),
    )

    assert response == {"job_id": "job_extract"}
    assert client.calls == [
        (
            "extract",
            {
                "function_id": "fn_123",
                "source_text": "hello",
                "options": {"num_results": 3},
            },
        )
    ]


def test_extract_module_accepts_metadata_ground_truth_and_extended_options() -> None:
    client = FakeWorkflowClient()

    response = extract(
        client,
        function_id="fn_123",
        source_text="hello",
        options=ExtractOptions(
            num_results=3,
            temperature=0.2,
            prompt_override="Use this prompt",
            extraction_model="openai/gpt-5.4-mini",
            llm_config_id="cfg_123",
            run_baseline_extraction=True,
        ),
        metadata={"document_id": "doc_123"},
        ground_truth={"field": "value"},
    )

    assert response == {"job_id": "job_extract"}
    assert client.calls == [
        (
            "extract",
            {
                "function_id": "fn_123",
                "source_text": "hello",
                "options": {
                    "num_results": 3,
                    "temperature": 0.2,
                    "prompt_override": "Use this prompt",
                    "extraction_model": "openai/gpt-5.4-mini",
                    "llm_config_id": "cfg_123",
                    "run_baseline_extraction": True,
                },
                "metadata": {"document_id": "doc_123"},
                "ground_truth": {"field": "value"},
            },
        )
    ]


def test_client_accepts_workflow_request_objects() -> None:
    transport = FakeTransport(
        [
            ResponseEnvelope(status_code=201, data={"config_id": "cfg_123"}),
            ResponseEnvelope(status_code=202, data={"job_id": "job_opt"}),
            ResponseEnvelope(status_code=202, data={"job_id": "job_compile"}),
            ResponseEnvelope(status_code=202, data={"job_id": "job_extract"}),
        ]
    )
    client = StabilizerClient(api_key="sk_test", transport=transport)

    config_response = client.create_llm_config(
            LLMConfigRequest(
                name="Primary config",
                provider="openai",
                api_key="provider-key",
                default_model="google/gemini-2.5-flash-lite",
                is_default=True,
                byok=True,
            )
    )
    optimize_response = client.optimize_prompt(
        OptimizeRequest(
            prompt="Extract event details",
            json_structure={"event_title": "string"},
            training_data=[
                TrainingExample(
                    source_text="Source",
                    extracted_json={"event_title": "Value"},
                )
            ],
        )
    )
    compile_response = client.compile_function(
        CompileRequest(
            name="Event details extractor",
            description="Extracts event details from text",
            tags=["events", "walkthrough"],
            prompt="Extract event details",
            json_structure={"event_title": "string"},
            training_data=[
                TrainingExample(
                    source_text="Source",
                    extracted_json={"event_title": "Value"},
                )
            ],
            grounding_methods=["hard_grounding", "constraints_validation"],
            compile_options=CompileOptions(num_prompt_variations=3),
        )
    )
    extract_response = client.extract(
        ExtractRequest(
            function_id="fn_123",
            source_text="hello",
            options=ExtractOptions(
                num_results=3,
                prompt_override="Use this prompt",
                run_baseline_extraction=True,
            ),
            metadata={"document_id": "doc_123"},
            ground_truth={"field": "value"},
        )
    )

    assert config_response == {"config_id": "cfg_123"}
    assert optimize_response == {"job_id": "job_opt"}
    assert compile_response == {"job_id": "job_compile"}
    assert extract_response == {"job_id": "job_extract"}
    assert [request.json_body for request in transport.requests] == [
        {
            "name": "Primary config",
            "provider": "openai",
            "api_key": "provider-key",
            "default_model": "google/gemini-2.5-flash-lite",
            "is_default": True,
            "byok": True,
        },
        {
            "prompt": "Extract event details",
            "json_structure": {"event_title": "string"},
            "training_data": [
                {
                    "source_text": "Source",
                    "extracted_json": {"event_title": "Value"},
                }
            ],
        },
        {
            "name": "Event details extractor",
            "description": "Extracts event details from text",
            "tags": ["events", "walkthrough"],
            "prompt": "Extract event details",
            "json_structure": {"event_title": "string"},
            "training_data": [
                {
                    "source_text": "Source",
                    "extracted_json": {"event_title": "Value"},
                }
            ],
            "grounding_methods": ["hard_grounding", "constraints_validation"],
            "compile_options": {"num_prompt_variations": 3},
        },
        {
            "function_id": "fn_123",
            "source_text": "hello",
            "options": {
                "num_results": 3,
                "prompt_override": "Use this prompt",
                "run_baseline_extraction": True,
            },
            "metadata": {"document_id": "doc_123"},
            "ground_truth": {"field": "value"},
        },
    ]

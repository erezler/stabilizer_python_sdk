from __future__ import annotations

import pytest

from stabilizer_python_sdk.compile import CompileOptions, compile_function
from stabilizer_python_sdk.config import LLMConfigRequest, create_llm_config
from stabilizer_python_sdk.extract import ExtractOptions


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def create_llm_config(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("create_llm_config", payload))
        return {"config_id": "cfg_123"}

    def compile_function(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("compile_function", payload))
        return {"job_id": "job_compile"}


def test_compile_options_emits_compile_strength() -> None:
    assert CompileOptions(compile_strength="high").as_payload() == {"compile_strength": "high"}


def test_compile_options_from_payload_reads_compile_strength() -> None:
    options = CompileOptions.from_payload({"compile_strength": "max"})
    assert options.compile_strength == "max"


def test_compile_options_rejects_legacy_compile_model_keyword() -> None:
    with pytest.raises(TypeError):
        CompileOptions(compile_model="openai/gpt-5.4")  # type: ignore[call-arg]


def test_extract_options_emits_grounding_strength() -> None:
    assert (
        ExtractOptions(grounding_strength="high").as_payload()["grounding_strength"] == "high"
    )


def test_extract_options_roundtrips_extraction_model_and_grounding_strength_independently() -> None:
    options = ExtractOptions(extraction_model="openai/gpt-5.4-mini", grounding_strength="low")
    payload = options.as_payload()
    assert payload == {
        "extraction_model": "openai/gpt-5.4-mini",
        "grounding_strength": "low",
    }
    restored = ExtractOptions.from_payload(payload)
    assert restored.extraction_model == "openai/gpt-5.4-mini"
    assert restored.grounding_strength == "low"


def test_extract_options_emits_grounding_methods_and_minimal_run() -> None:
    payload = ExtractOptions(
        grounding_methods=["hard_grounding", "coverage_check"],
        minimal_run=True,
    ).as_payload()
    assert payload == {
        "grounding_methods": ["hard_grounding", "coverage_check"],
        "minimal_run": True,
    }


def test_extract_options_omits_grounding_methods_and_minimal_run_when_unset() -> None:
    assert ExtractOptions(num_results=3).as_payload() == {"num_results": 3}


def test_extract_options_roundtrips_grounding_methods_and_minimal_run() -> None:
    restored = ExtractOptions.from_payload(
        {"grounding_methods": ["stress_test"], "minimal_run": False}
    )
    assert restored.grounding_methods == ["stress_test"]
    assert restored.minimal_run is False


def test_extract_options_from_payload_ignores_non_list_grounding_methods() -> None:
    assert ExtractOptions.from_payload({"grounding_methods": "hard_grounding"}).grounding_methods is None


def test_llm_config_request_emits_compile_strength() -> None:
    payload = LLMConfigRequest(name="primary", compile_strength="medium").as_payload()
    assert payload == {"name": "primary", "compile_strength": "medium"}


def test_llm_config_request_roundtrips_compile_strength() -> None:
    restored = LLMConfigRequest.from_payload({"name": "primary", "compile_strength": "max"})
    assert restored.compile_strength == "max"


def test_create_llm_config_helper_forwards_compile_strength() -> None:
    client = _RecordingClient()

    create_llm_config(client, name="primary", compile_strength="high")

    assert client.calls == [
        ("create_llm_config", {"name": "primary", "compile_strength": "high"}),
    ]


def test_compile_function_payload_includes_compile_strength_and_never_compile_model() -> None:
    client = _RecordingClient()

    compile_function(
        client,
        prompt="extract me",
        json_structure={"x": "string"},
        compile_options=CompileOptions(compile_strength="max"),
    )

    assert client.calls == [
        (
            "compile_function",
            {
                "prompt": "extract me",
                "json_structure": {"x": "string"},
                "compile_options": {"compile_strength": "max"},
            },
        )
    ]
    serialized_compile = client.calls[0][1]["compile_options"]
    assert isinstance(serialized_compile, dict)
    assert "compile_model" not in serialized_compile

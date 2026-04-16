from __future__ import annotations

import json
from pathlib import Path


def test_compile_payload_file_exists_and_contains_minimum_compile_fields() -> None:
    payload = json.loads(Path("compile-input.json").read_text(encoding="utf-8"))
    compile_options = payload["compile_options"]

    assert payload["name"] == "Event details extractor"
    assert isinstance(payload["prompt"], str)
    assert isinstance(payload["json_structure"], dict)
    assert isinstance(payload["training_data"], list)
    assert isinstance(compile_options, dict)
    assert (
        compile_options.get("num_prompt_variations") == 3
        or (
            isinstance(compile_options.get("optimized_prompts"), list)
            and len(compile_options["optimized_prompts"]) > 0
            and all(isinstance(prompt, str) and prompt for prompt in compile_options["optimized_prompts"])
        )
    )


def test_extract_payload_file_exists_and_contains_minimum_extract_fields() -> None:
    payload = json.loads(Path("extract-input.json").read_text(encoding="utf-8"))

    assert payload["function_id"] == "fn_replace_me"
    assert isinstance(payload["source_text"], str)
    assert payload["options"]["num_results"] == 3


def test_compile_heavy_payload_file_exists_and_contains_contract_fixture_fields() -> None:
    payload = json.loads(Path("compile-heavy-input.json").read_text(encoding="utf-8"))

    assert payload["name"] == "Contract field extractor"
    assert isinstance(payload["prompt"], str)
    assert isinstance(payload["json_structure"], dict)
    assert isinstance(payload["training_data"], list)
    assert payload["compile_options"]["num_prompt_variations"] == 3
    assert payload["json_structure"]["Document Name"] == []
    assert payload["json_structure"]["Third Party Beneficiary"] == []


def test_extract_heavy_payload_file_exists_and_contains_contract_fixture_fields() -> None:
    payload = json.loads(Path("extract-heavy-input.json").read_text(encoding="utf-8"))

    assert payload["function_id"] == "fn_replace_me"
    assert isinstance(payload["source_text"], str)
    assert payload["options"]["num_results"] == 3
    assert "MASTER SUPPLY AGREEMENT" in payload["source_text"]

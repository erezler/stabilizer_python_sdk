from __future__ import annotations

import json
from pathlib import Path


def test_compile_payload_file_exists_and_contains_minimum_compile_fields() -> None:
    payload = json.loads(Path("compile.json").read_text(encoding="utf-8"))

    assert payload["name"] == "Event details extractor"
    assert isinstance(payload["prompt"], str)
    assert isinstance(payload["json_structure"], dict)
    assert isinstance(payload["training_data"], list)
    assert payload["compile_options"]["num_prompt_variations"] == 3


def test_extract_payload_file_exists_and_contains_minimum_extract_fields() -> None:
    payload = json.loads(Path("extract.json").read_text(encoding="utf-8"))

    assert payload["function_id"] == "fn_replace_me"
    assert isinstance(payload["source_text"], str)
    assert payload["options"]["num_results"] == 3

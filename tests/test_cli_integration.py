from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from stabilizer_python_sdk import StabilizerClient

INTEGRATION_ENABLE_ENV = "STABILIZER_RUN_INTEGRATION"
API_KEY_ENV = "STABILIZER_API_KEY"
PROVIDER_API_KEY_ENV = "STABILIZER_PROVIDER_API_KEY"
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_cli_json_output(output: str) -> object:
    stripped = output.strip()
    if not stripped:
        return None
    if stripped == "null":
        return None

    for marker in ("\n{\n", "\n[\n"):
        marker_index = output.find(marker)
        if marker_index != -1:
            return json.loads(output[marker_index + 1 :])

    return json.loads(output)


def _write_sequence_progress(command_name: str, *, status: str, stream: io.TextIOBase) -> None:
    status_color = {
        "running": "\x1b[34m",
        "passed": "\x1b[32m",
        "failed": "\x1b[31m",
    }.get(status, "\x1b[37m")
    stream.write(
        "\x1b[36m[integration]\x1b[0m "
        f"\x1b[33m{command_name}\x1b[0m "
        f"{status_color}{status}\x1b[0m\n"
    )
    stream.flush()


def _extract_function_id(payload: object) -> str:
    if not isinstance(payload, dict):
        raise AssertionError(f"Expected a dict payload, got: {payload!r}")
    function_id = payload.get("function_id")
    if isinstance(function_id, str) and function_id:
        return function_id
    result = payload.get("result")
    if isinstance(result, dict):
        nested_function_id = result.get("function_id")
        if isinstance(nested_function_id, str) and nested_function_id:
            return nested_function_id
    raise AssertionError(f"Could not find function_id in payload: {payload!r}")


def _extract_optimized_prompt(payload: object) -> str:
    if not isinstance(payload, dict):
        raise AssertionError(f"Expected a dict payload, got: {payload!r}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise AssertionError(f"Optimize payload is missing result: {payload!r}")
    optimized_prompt = result.get("optimized_prompt")
    if isinstance(optimized_prompt, str) and optimized_prompt:
        return optimized_prompt
    optimized_prompts = result.get("optimized_prompts")
    if isinstance(optimized_prompts, list):
        for candidate in optimized_prompts:
            if isinstance(candidate, str) and candidate:
                return candidate
    raise AssertionError(f"Optimize payload is missing optimized prompt data: {payload!r}")


def _require_live_cli_env(*, require_provider_api_key: bool) -> dict[str, str]:
    if os.getenv(INTEGRATION_ENABLE_ENV) != "1":
        pytest.skip(f"Set {INTEGRATION_ENABLE_ENV}=1 to enable live CLI integration tests.")

    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        pytest.skip(f"Set {API_KEY_ENV} to enable live CLI integration tests.")

    if require_provider_api_key and not os.getenv(PROVIDER_API_KEY_ENV):
        pytest.skip(f"Set {PROVIDER_API_KEY_ENV} to enable live config/create integration tests.")

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_DIR) if not existing_pythonpath else os.pathsep.join([str(SRC_DIR), existing_pythonpath])
    return env


def _run_cli_command(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float = 900.0,
) -> tuple[int, object, str, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "stabilizer_python_sdk", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        check=False,
    )
    parsed_output = _parse_cli_json_output(completed.stdout)
    return completed.returncode, parsed_output, completed.stdout, completed.stderr


def test_write_sequence_progress_uses_ansi_colors() -> None:
    stream = io.StringIO()

    _write_sequence_progress("health", status="running", stream=stream)
    _write_sequence_progress("health", status="passed", stream=stream)

    assert stream.getvalue() == (
        "\x1b[36m[integration]\x1b[0m "
        "\x1b[33mhealth\x1b[0m "
        "\x1b[34mrunning\x1b[0m\n"
        "\x1b[36m[integration]\x1b[0m "
        "\x1b[33mhealth\x1b[0m "
        "\x1b[32mpassed\x1b[0m\n"
    )


@pytest.mark.integration
def test_live_cli_public_commands_round_trip(tmp_path: Path) -> None:
    env = _require_live_cli_env(require_provider_api_key=False)

    exit_code, health_result, stdout, stderr = _run_cli_command(["health"], cwd=tmp_path, env=env)
    assert exit_code == 0, f"health failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    assert isinstance(health_result, dict)
    assert str(health_result.get("status", "")).lower() in {"ok", "healthy"}

    exit_code, models_result, stdout, stderr = _run_cli_command(["models"], cwd=tmp_path, env=env)
    assert exit_code == 0, f"models failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    assert isinstance(models_result, dict)
    models = models_result.get("models")
    assert isinstance(models, list)
    assert models


@pytest.mark.integration
def test_live_cli_sequence_round_trips_to_server(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = _require_live_cli_env(require_provider_api_key=True)
    api_key = env[API_KEY_ENV]
    cleanup_client = StabilizerClient(api_key=api_key)
    unique_suffix = uuid4().hex[:8]
    compile_name = f"CLI Integration {unique_suffix}"
    updated_function_name = f"{compile_name} Updated"
    function_tag = f"it-{unique_suffix}"
    usage_to = date.today() - timedelta(days=1)
    usage_from = usage_to - timedelta(days=14)

    created_api_key_id: str | None = None
    created_config_id: str | None = None
    compiled_function_id: str | None = None

    def run_command(args: list[str]) -> object:
        command_name = args[0]
        with capsys.disabled():
            _write_sequence_progress(command_name, status="running", stream=sys.stderr)
        exit_code, payload, stdout, stderr = _run_cli_command(args, cwd=tmp_path, env=env)
        with capsys.disabled():
            _write_sequence_progress(
                command_name,
                status="passed" if exit_code == 0 else "failed",
                stream=sys.stderr,
            )
        assert exit_code == 0, f"{command_name} failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
        return payload

    try:
        org_result = run_command(["org", "--api-key", api_key])
        assert isinstance(org_result, dict)
        assert isinstance(org_result.get("org_id"), str)

        org_name = org_result.get("name")
        assert isinstance(org_name, str)
        _write_json(tmp_path / "org-update.json", {"name": org_name})

        api_keys_result = run_command(["api-keys", "--api-key", api_key])
        assert isinstance(api_keys_result, list)

        _write_json(
            tmp_path / "api-key-create.json",
            {"name": f"CLI integration key {unique_suffix}", "scope": "read_only"},
        )
        api_key_create_result = run_command(
            ["api-key-create", "--api-key", api_key, "--payload-file", ".\\api-key-create.json"]
        )
        assert isinstance(api_key_create_result, dict)
        created_api_key_id = str(api_key_create_result["key_id"])
        assert created_api_key_id

        org_update_result = run_command(
            ["org-update", "--api-key", api_key, "--payload-file", ".\\org-update.json"]
        )
        assert isinstance(org_update_result, dict)
        assert str(org_update_result.get("org_id")) == str(org_result["org_id"])

        configs_result = run_command(["configs", "--api-key", api_key])
        assert isinstance(configs_result, list)

        _write_json(
            tmp_path / "config-input.json",
            {
                "name": f"CLI integration config {unique_suffix}",
                "provider": "openrouter",
                "default_model": "google/gemini-2.5-flash-lite",
                "is_default": False,
                "byok": True,
            },
        )
        config_result = run_command(["config", "--api-key", api_key, "--payload-file", ".\\config-input.json"])
        assert isinstance(config_result, dict)
        created_config_id = str(config_result["config_id"])
        assert created_config_id

        _write_json(tmp_path / "config-update.json", {"name": f"CLI integration config updated {unique_suffix}"})
        config_update_result = run_command(
            [
                "config-update",
                "--api-key",
                api_key,
                "--config",
                created_config_id,
                "--payload-file",
                ".\\config-update.json",
            ]
        )
        assert isinstance(config_update_result, dict)
        assert str(config_update_result.get("config_id")) == created_config_id

        _write_json(
            tmp_path / "optimize-input.json",
            {
                "prompt": "Extract the invoice total and due date into JSON.",
                "json_structure": {
                    "invoice_total": "number",
                    "due_date": "string",
                },
                "training_data": [
                    {
                        "source_text": "Invoice INV-10 is due on 2026-05-01 and totals $125.",
                        "extracted_json": {"invoice_total": 125, "due_date": "2026-05-01"},
                    }
                ],
            },
        )
        _write_json(
            tmp_path / "compile-input.json",
            {
                "name": compile_name,
                "description": "CLI integration test function",
                "tags": [function_tag, "integration"],
                "prompt": "Extract the invoice total and due date into JSON.",
                "json_structure": {
                    "invoice_total": "number",
                    "due_date": "string",
                },
                "training_data": [
                    {
                        "source_text": "Invoice INV-10 is due on 2026-05-01 and totals $125.",
                        "extracted_json": {"invoice_total": 125, "due_date": "2026-05-01"},
                    }
                ],
                "compile_options": {
                    "num_prompt_variations": 3,
                },
            },
        )

        optimize_result = run_command(
            [
                "optimize",
                "--api-key",
                api_key,
                "--payload-file",
                ".\\optimize-input.json",
                "--config",
                created_config_id,
                "--poll",
                "--alter-compile",
                ".\\compile-input.json",
            ]
        )
        assert isinstance(optimize_result, dict)
        assert str(optimize_result.get("status", "")).lower() == "completed"
        optimize_job_id = str(optimize_result["job_id"])

        altered_compile_payload = json.loads((tmp_path / "compile-input.json").read_text(encoding="utf-8"))
        assert altered_compile_payload["prompt"] == _extract_optimized_prompt(optimize_result)

        compile_result = run_command(
            [
                "compile",
                "--api-key",
                api_key,
                "--payload-file",
                ".\\compile-input.json",
                "--config",
                created_config_id,
                "--poll",
            ]
        )
        assert isinstance(compile_result, dict)
        assert str(compile_result.get("status", "")).lower() == "completed"
        compile_job_id = str(compile_result["job_id"])
        compiled_function_id = _extract_function_id(compile_result)

        functions_result = run_command(
            ["functions", "--api-key", api_key, "--name", compile_name, "--tag", function_tag]
        )
        assert isinstance(functions_result, list)

        function_result = run_command(["function", "--api-key", api_key, "--function", compiled_function_id])
        assert isinstance(function_result, dict)
        assert str(function_result.get("function_id")) == compiled_function_id

        _write_json(tmp_path / "function-update.json", {"name": updated_function_name})
        function_update_result = run_command(
            [
                "function-update",
                "--api-key",
                api_key,
                "--function",
                compiled_function_id,
                "--payload-file",
                ".\\function-update.json",
            ]
        )
        assert isinstance(function_update_result, dict)
        assert str(function_update_result.get("function_id")) == compiled_function_id

        _write_json(
            tmp_path / "extract-input.json",
            {
                "function_id": "fn_replace_me",
                "source_text": "Invoice INV-11 is due on 2026-06-15 and totals $214.",
                "options": {"num_results": 1},
            },
        )
        extract_result = run_command(
            [
                "extract",
                "--api-key",
                api_key,
                "--payload-file",
                ".\\extract-input.json",
                "--function",
                compiled_function_id,
                "--poll",
            ]
        )
        assert isinstance(extract_result, dict)
        assert str(extract_result.get("status", "")).lower() == "completed"
        extract_job_id = str(extract_result["job_id"])

        extractions_result = run_command(["extractions", "--api-key", api_key])
        assert isinstance(extractions_result, list)

        job_result = run_command(["job", "--api-key", api_key, "--job", extract_job_id])
        assert isinstance(job_result, dict)
        assert str(job_result.get("job_id")) == extract_job_id

        poll_result = run_command(["poll", "--api-key", api_key, "--job", extract_job_id, "--timeout", "600"])
        assert isinstance(poll_result, dict)
        assert str(poll_result.get("job_id")) == extract_job_id
        assert str(poll_result.get("status", "")).lower() == "completed"

        usage_result = run_command(
            [
                "usage",
                "--api-key",
                api_key,
                "--from",
                usage_from.isoformat(),
                "--to",
                usage_to.isoformat(),
            ]
        )
        assert isinstance(usage_result, dict)
        assert "org_id" in usage_result

        state_result = run_command(["state", "latest"])
        assert state_result == {
            "state_file": "general",
            "config_id": created_config_id,
            "optimize_job_id": optimize_job_id,
            "compile_job_id": compile_job_id,
            "function_id": compiled_function_id,
            "extract_job_id": extract_job_id,
        }

        revoke_result = run_command(["api-key-revoke", "--api-key", api_key, "--key", created_api_key_id])
        assert revoke_result is None
        created_api_key_id = None

        function_delete_result = run_command(
            ["function-delete", "--api-key", api_key, "--function", compiled_function_id]
        )
        assert function_delete_result is None
        compiled_function_id = None

        config_delete_result = run_command(["config-delete", "--api-key", api_key, "--config", created_config_id])
        assert config_delete_result is None
        created_config_id = None
    finally:
        with contextlib.suppress(Exception):
            if compiled_function_id:
                cleanup_client.delete_function(compiled_function_id)
        with contextlib.suppress(Exception):
            if created_config_id:
                cleanup_client.delete_llm_config(created_config_id)
        with contextlib.suppress(Exception):
            if created_api_key_id:
                cleanup_client.revoke_api_key(created_api_key_id)

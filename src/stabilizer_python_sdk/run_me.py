from __future__ import annotations

import argparse
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stabilizer_python_sdk.client import DEFAULT_BASE_URL, StabilizerClient
from stabilizer_python_sdk.compile import CompileRequest, run_compile_step
from stabilizer_python_sdk.config import LLMConfigRequest, run_config_step
from stabilizer_python_sdk.extract import ExtractRequest, run_extract_step
from stabilizer_python_sdk.optimize import OptimizeRequest, run_optimize_step
from stabilizer_python_sdk.workflow_runtime import (
    WorkflowConsole,
    default_now_provider,
    load_json_object,
    load_state,
    resolve_state_file,
)

# Developer-editable workflow settings.
DEFAULT_API_KEY = os.getenv("STABILIZER_API_KEY", "YOUR_STABILIZER_API_KEY")
DEFAULT_PROVIDER_API_KEY = os.getenv("STABILIZER_PROVIDER_API_KEY", "YOUR_PROVIDER_API_KEY")
DEFAULT_TEMP_DB_DIR = Path("temp_db")
DEFAULT_STATE_FILE: str | Path | None = None
DEFAULT_COMPILE_PAYLOAD_FILE = Path("compile.json")
DEFAULT_EXTRACT_PAYLOAD_FILE = Path("extract.json")
DEFAULT_POLL_INTERVAL = 2.0
DEFAULT_POLL_TIMEOUT = 600.0
DEFAULT_CONFIG_REQUEST = LLMConfigRequest(
    name="Primary config",
    provider="openai",
    api_key=DEFAULT_PROVIDER_API_KEY,
    default_model="openai/gpt-5.4-mini",
    is_default=True,
)


@dataclass(frozen=True)
class RunMeSettings:
    api_key: str = DEFAULT_API_KEY
    base_url: str = DEFAULT_BASE_URL
    temp_db_dir: Path = DEFAULT_TEMP_DB_DIR
    state_file: str | Path | None = DEFAULT_STATE_FILE
    config_request: LLMConfigRequest = DEFAULT_CONFIG_REQUEST
    optimize_request: OptimizeRequest | None = None
    compile_payload_file: Path = DEFAULT_COMPILE_PAYLOAD_FILE
    extract_payload_file: Path = DEFAULT_EXTRACT_PAYLOAD_FILE
    poll_interval: float = DEFAULT_POLL_INTERVAL
    poll_timeout: float = DEFAULT_POLL_TIMEOUT


def run_all(
    *,
    settings: RunMeSettings = RunMeSettings(),
    client: StabilizerClient | None = None,
    console: WorkflowConsole | None = None,
    now_provider=default_now_provider,
    sleeper=time.sleep,
    monotonic=time.monotonic,
) -> dict[str, Any]:
    workflow_console = console or WorkflowConsole()
    state_path = resolve_state_file(
        state_file=settings.state_file,
        temp_db_dir=settings.temp_db_dir,
        now_provider=now_provider,
    )
    workflow_console.info("run", f"Using state file {state_path}.")

    active_client = client or _make_client(settings)
    optimize_request = settings.optimize_request or load_optimize_request(settings.compile_payload_file)
    compile_request = load_compile_request(settings.compile_payload_file)
    extract_request = load_extract_request(settings.extract_payload_file)

    run_config_step(
        active_client,
        request=settings.config_request,
        state_file=state_path,
        temp_db_dir=settings.temp_db_dir,
        console=workflow_console,
        now_provider=now_provider,
    )
    run_optimize_step(
        active_client,
        request=optimize_request,
        state_file=state_path,
        temp_db_dir=settings.temp_db_dir,
        console=workflow_console,
        poll_interval=settings.poll_interval,
        timeout=settings.poll_timeout,
        now_provider=now_provider,
        sleeper=sleeper,
        monotonic=monotonic,
    )
    run_compile_step(
        active_client,
        request=compile_request,
        state_file=state_path,
        temp_db_dir=settings.temp_db_dir,
        console=workflow_console,
        poll_interval=settings.poll_interval,
        timeout=settings.poll_timeout,
        now_provider=now_provider,
        sleeper=sleeper,
        monotonic=monotonic,
    )
    run_extract_step(
        active_client,
        request=extract_request,
        state_file=state_path,
        temp_db_dir=settings.temp_db_dir,
        console=workflow_console,
        poll_interval=settings.poll_interval,
        timeout=settings.poll_timeout,
        now_provider=now_provider,
        sleeper=sleeper,
        monotonic=monotonic,
    )

    _, state = load_state(
        state_file=state_path,
        temp_db_dir=settings.temp_db_dir,
        now_provider=now_provider,
    )
    workflow_console.success("run", "Workflow finished.")
    return state


def load_optimize_request(path: str | Path) -> OptimizeRequest:
    payload = load_json_object(path)
    optimize_payload = {
        "prompt": payload["prompt"],
        "json_structure": payload["json_structure"],
        "training_data": payload.get("training_data", []),
    }
    return OptimizeRequest.from_payload(optimize_payload)


def load_compile_request(path: str | Path) -> CompileRequest:
    return CompileRequest.from_payload(load_json_object(path))


def load_extract_request(path: str | Path) -> ExtractRequest:
    return ExtractRequest.from_payload(load_json_object(path))


def _make_client(settings: RunMeSettings) -> StabilizerClient:
    if not settings.api_key or settings.api_key == "YOUR_STABILIZER_API_KEY":
        raise ValueError("Set STABILIZER_API_KEY or override RunMeSettings.api_key before running.")
    return StabilizerClient(api_key=settings.api_key, base_url=settings.base_url)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m stabilizer_python_sdk.run_me",
        description="Run the config -> optimize -> compile -> extract workflow.",
    )
    parser.add_argument("--state-file", help="Optional path to an existing state JSON file.")
    parser.add_argument(
        "--temp-db-dir",
        default=str(DEFAULT_TEMP_DB_DIR),
        help="Directory that stores cumulative workflow state files.",
    )
    parser.add_argument(
        "--compile-payload-file",
        default=str(DEFAULT_COMPILE_PAYLOAD_FILE),
        help="JSON payload file for compile and optimize requests.",
    )
    parser.add_argument(
        "--extract-payload-file",
        default=str(DEFAULT_EXTRACT_PAYLOAD_FILE),
        help="JSON payload file for the extract request.",
    )
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Stabilizer API key.")
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL)
    parser.add_argument("--poll-timeout", type=float, default=DEFAULT_POLL_TIMEOUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    parsed = parser.parse_args(list(argv) if argv is not None else None)
    run_all(
        settings=RunMeSettings(
            api_key=parsed.api_key,
            temp_db_dir=Path(parsed.temp_db_dir),
            state_file=parsed.state_file,
            compile_payload_file=Path(parsed.compile_payload_file),
            extract_payload_file=Path(parsed.extract_payload_file),
            poll_interval=parsed.poll_interval,
            poll_timeout=parsed.poll_timeout,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

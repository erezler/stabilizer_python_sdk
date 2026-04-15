from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from stabilizer_python_sdk import ApiError, StabilizerClient
from stabilizer_python_sdk.config import LLMConfigRequest

DEFAULT_TEMP_DB_DIR = Path("temp_db")
DEFAULT_CONFIG_FILE = Path("config.json")
DEFAULT_OPTIMIZE_FILE = Path("optimize.json")
DEFAULT_COMPILE_FILE = Path("compile.json")
DEFAULT_EXTRACT_FILE = Path("extract.json")
GENERAL_STATE_SUBDIR = "general"
RUN_ME_STATE_SUBDIR = "run_me"
DEFAULT_POLL_INTERVAL = 2.0
TERMINAL_JOB_STATUSES = frozenset({"completed", "failed"})


def _load_env_file(path: str | Path = ".env.local") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = key.strip()
        if not normalized_key:
            continue
        normalized_value = value.strip()
        if (
            len(normalized_value) >= 2
            and normalized_value[0] == normalized_value[-1]
            and normalized_value[0] in {"'", '"'}
        ):
            normalized_value = normalized_value[1:-1]
        os.environ.setdefault(normalized_key, normalized_value)


def _default_api_key() -> str | None:
    return os.getenv("STABILIZER_API_KEY")


def _require_api_key(api_key: str | None) -> str:
    resolved_api_key = api_key or _default_api_key()
    if resolved_api_key:
        return resolved_api_key
    raise ValueError("Missing API key. Pass --api-key or set STABILIZER_API_KEY in the environment or .env.local.")


_load_env_file()


def _make_client(api_key: str | None = None) -> StabilizerClient:
    return StabilizerClient(api_key=api_key)


def _read_payload(path: str) -> dict[str, Any]:
    payload_text = Path(path).read_text(encoding="utf-8")
    payload = json.loads(payload_text)
    if not isinstance(payload, dict):
        raise ValueError("Payload file must contain a JSON object.")
    return payload


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2))


def _read_state(path: Path) -> dict[str, Any]:
    return _read_payload(str(path))


def _run_me_state_dir(temp_db_dir: Path) -> Path:
    return temp_db_dir / RUN_ME_STATE_SUBDIR


def _general_dir(temp_db_dir: Path) -> Path:
    return temp_db_dir / GENERAL_STATE_SUBDIR


def _general_file(temp_db_dir: Path, filename: str | Path) -> Path:
    return _general_dir(temp_db_dir) / Path(filename).name


def _saved_request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    request = payload.get("_request")
    if isinstance(request, dict):
        return request
    return payload


def _payload_has_keys(payload: dict[str, Any], required_keys: set[str]) -> bool:
    return required_keys.issubset(payload.keys())


def _load_saved_payload_candidate(
    path: Path,
    *,
    required_keys: set[str],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = _saved_request_payload(_read_payload(str(path)))
    if _payload_has_keys(payload, required_keys):
        return payload
    return None


def _load_saved_response_candidate(
    path: Path,
    *,
    required_keys: set[str],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = _read_payload(str(path))
    if _payload_has_keys(payload, required_keys):
        return payload
    return None


def _extract_config_id(payload: dict[str, Any]) -> str | None:
    config_id = payload.get("config_id")
    if config_id is None:
        return None
    return str(config_id)


def _extract_function_id(payload: dict[str, Any]) -> str | None:
    function_id = payload.get("function_id")
    if function_id is not None:
        return str(function_id)
    result = payload.get("result")
    if isinstance(result, dict):
        nested_function_id = result.get("function_id")
        if nested_function_id is not None:
            return str(nested_function_id)
    return None


def _load_id_from_saved_file(
    path: Path,
    *,
    extractor,
) -> str | None:
    if not path.is_file():
        return None
    return extractor(_read_payload(str(path)))


def _resolve_config_id(config_id: str | None, temp_db_dir: Path) -> str | None:
    if config_id is not None:
        return config_id
    saved_config_id = _load_id_from_saved_file(
        _general_file(temp_db_dir, DEFAULT_CONFIG_FILE),
        extractor=_extract_config_id,
    )
    if saved_config_id is not None:
        return saved_config_id
    return _load_id_from_saved_file(DEFAULT_CONFIG_FILE, extractor=_extract_config_id)


def _resolve_function_id(function_id: str | None, temp_db_dir: Path) -> str | None:
    if function_id is not None:
        return function_id
    saved_function_id = _load_id_from_saved_file(
        _general_file(temp_db_dir, DEFAULT_COMPILE_FILE),
        extractor=_extract_function_id,
    )
    if saved_function_id is not None:
        return saved_function_id
    return _load_id_from_saved_file(DEFAULT_COMPILE_FILE, extractor=_extract_function_id)


def _default_config_payload_from_env() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": os.getenv("STABILIZER_CONFIG_NAME", "Primary config"),
        "provider": os.getenv("STABILIZER_PROVIDER", "openai"),
        "default_model": os.getenv("STABILIZER_DEFAULT_MODEL", "google/gemini-2.5-flash-lite"),
        "is_default": os.getenv("STABILIZER_IS_DEFAULT", "true").strip().lower() in {"1", "true", "yes", "on"},
        "byok": os.getenv("STABILIZER_BYOK", "false").strip().lower() in {"1", "true", "yes", "on"},
    }
    provider_api_key = os.getenv("STABILIZER_PROVIDER_API_KEY")
    if provider_api_key:
        payload["api_key"] = provider_api_key
    return payload


def _load_config_payload(payload_file: str | None, temp_db_dir: Path) -> dict[str, Any]:
    if payload_file is not None:
        return _read_payload(payload_file)
    general_payload = _load_saved_payload_candidate(
        _general_file(temp_db_dir, DEFAULT_CONFIG_FILE),
        required_keys={"name", "provider", "default_model"},
    )
    if general_payload is not None:
        return general_payload
    if DEFAULT_CONFIG_FILE.is_file():
        return _read_payload(str(DEFAULT_CONFIG_FILE))
    return _default_config_payload_from_env()


def _load_module_payload(
    payload_file: str | None,
    *,
    temp_db_dir: Path,
    default_file: Path,
    required_keys: set[str],
) -> dict[str, Any]:
    if payload_file is not None:
        return _read_payload(payload_file)
    general_payload = _load_saved_payload_candidate(
        _general_file(temp_db_dir, default_file),
        required_keys=required_keys,
    )
    if general_payload is not None:
        return general_payload
    if default_file.is_file():
        return _read_payload(str(default_file))
    raise ValueError(f"Payload file '{default_file}' was not found.")


def _save_general_payload(
    temp_db_dir: Path,
    filename: str | Path,
    *,
    request: dict[str, Any],
    response: dict[str, Any],
) -> None:
    target = _general_file(temp_db_dir, filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(response)
    payload["_request"] = request
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_polled_job_result(temp_db_dir: Path, job: dict[str, Any]) -> None:
    job_type = str(job.get("type", "")).lower()
    target_file: Path | None = None
    if job_type == "compile":
        target_file = _general_file(temp_db_dir, DEFAULT_COMPILE_FILE)
    elif job_type == "extract":
        target_file = _general_file(temp_db_dir, DEFAULT_EXTRACT_FILE)

    if target_file is None:
        return

    payload = dict(job)
    if target_file.is_file():
        existing_payload = _read_payload(str(target_file))
        existing_request = existing_payload.get("_request")
        if isinstance(existing_request, dict):
            payload["_request"] = existing_request

    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _latest_state_file(temp_db_dir: Path) -> Path:
    state_dir = _run_me_state_dir(temp_db_dir)
    state_files = sorted(path for path in state_dir.glob("*.json") if path.is_file())
    if not state_files:
        raise ValueError(f"No state files found in '{state_dir}'.")
    return state_files[-1]


def _resolve_state_path(target: str, temp_db_dir: Path) -> Path:
    if target == "latest":
        return _latest_state_file(temp_db_dir)
    candidate = Path(target)
    if candidate.is_file():
        return candidate
    resolved = _run_me_state_dir(temp_db_dir) / target
    if resolved.is_file():
        return resolved
    raise ValueError(f"State file '{target}' was not found.")


def _summarize_state(path: Path) -> dict[str, Any]:
    state = _read_state(path)
    steps = state.get("steps", {})
    config_step = steps.get("config", {})
    optimize_step = steps.get("optimize", {})
    compile_step = steps.get("compile", {})
    extract_step = steps.get("extract", {})
    return {
        "state_file": path.name,
        "config_id": config_step.get("config_id"),
        "optimize_job_id": optimize_step.get("job_id"),
        "compile_job_id": compile_step.get("job_id"),
        "function_id": compile_step.get("function_id"),
        "extract_job_id": extract_step.get("job_id"),
    }


def _summarize_general_state(temp_db_dir: Path) -> dict[str, Any]:
    config_payload = _load_saved_response_candidate(
        _general_file(temp_db_dir, DEFAULT_CONFIG_FILE),
        required_keys={"config_id"},
    ) or {}
    optimize_payload = _load_saved_response_candidate(
        _general_file(temp_db_dir, DEFAULT_OPTIMIZE_FILE),
        required_keys={"job_id"},
    ) or {}
    compile_payload = _load_saved_response_candidate(
        _general_file(temp_db_dir, DEFAULT_COMPILE_FILE),
        required_keys={"job_id"},
    ) or {}
    extract_payload = _load_saved_response_candidate(
        _general_file(temp_db_dir, DEFAULT_EXTRACT_FILE),
        required_keys={"job_id"},
    ) or {}
    return {
        "state_file": GENERAL_STATE_SUBDIR,
        "config_id": config_payload.get("config_id"),
        "optimize_job_id": optimize_payload.get("job_id"),
        "compile_job_id": compile_payload.get("job_id"),
        "function_id": _extract_function_id(compile_payload),
        "extract_job_id": extract_payload.get("job_id"),
    }


def _load_request_payload(
    payload_file: str | None,
    *,
    temp_db_dir: Path,
    default_file: Path,
    required_keys: set[str],
    config_id: str | None = None,
    function_id: str | None = None,
) -> dict[str, Any]:
    payload = _load_module_payload(
        payload_file,
        temp_db_dir=temp_db_dir,
        default_file=default_file,
        required_keys=required_keys,
    )
    if config_id is not None:
        payload["config_id"] = config_id
    if function_id is not None:
        payload["function_id"] = function_id
    return payload


def _poll_job_with_progress(
    client: StabilizerClient,
    *,
    job_id: str,
    timeout: float,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_line_length = 0
    while True:
        job = client.get_job(job_id)
        status = str(job.get("status", "")).lower()
        progress = job.get("progress")
        if isinstance(progress, (int, float)):
            message = f"Progress: {int(progress)}% ({status or 'unknown'})"
        else:
            message = f"Progress: status={status or 'unknown'}"
        padded_message = message.ljust(last_line_length)
        sys.stdout.write(f"\r{padded_message}")
        sys.stdout.flush()
        last_line_length = len(message)
        if status in TERMINAL_JOB_STATUSES:
            sys.stdout.write("\n")
            sys.stdout.flush()
            return job
        if time.monotonic() >= deadline:
            sys.stdout.write("\n")
            sys.stdout.flush()
            raise TimeoutError(f"Timed out waiting for job '{job_id}'.")
        time.sleep(poll_interval)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stabilizer_python_sdk",
        description="CLI for the Stabilizer API.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("health", help="Run GET /v1/health.")
    subparsers.add_parser("models", help="Run GET /v1/supported-models.")

    config_parser = subparsers.add_parser(
        "config",
        help="Run POST /v1/llm-configs with a JSON payload file.",
    )
    config_parser.add_argument("--api-key", help="Stabilizer API key.")
    config_parser.add_argument(
        "--payload-file",
        default=None,
        help="Path to a JSON file for the config request body.",
    )

    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Run POST /v1/prompt-optimizations with a JSON payload file.",
    )
    optimize_parser.add_argument("--api-key", help="Stabilizer API key.")
    optimize_parser.add_argument(
        "--payload-file",
        default=None,
        help="Path to a JSON file for the optimize request body.",
    )
    optimize_parser.add_argument(
        "--config",
        help="Existing config_id to inject into the optimize payload.",
    )
    optimize_parser.add_argument(
        "--wait",
        action="store_true",
        help="Poll the returned job until completion.",
    )
    optimize_parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Maximum seconds to wait when --wait is used.",
    )

    compile_parser = subparsers.add_parser(
        "compile",
        help="Run POST /v1/functions with a JSON payload file.",
    )
    compile_parser.add_argument("--api-key", help="Stabilizer API key.")
    compile_parser.add_argument(
        "--payload-file",
        default=None,
        help="Path to a JSON file for the compile request body.",
    )
    compile_parser.add_argument(
        "--config",
        help="Existing config_id to inject into the compile payload.",
    )
    compile_parser.add_argument(
        "--wait",
        action="store_true",
        help="Poll the returned job until completion.",
    )
    compile_parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Maximum seconds to wait when --wait is used.",
    )

    extract_parser = subparsers.add_parser(
        "extract",
        help="Run POST /v1/extract with a JSON payload file.",
    )
    extract_parser.add_argument("--api-key", help="Stabilizer API key.")
    extract_parser.add_argument(
        "--payload-file",
        default=None,
        help="Path to a JSON file for the extraction request body.",
    )
    extract_parser.add_argument(
        "--function",
        help="Existing function_id to inject into the extraction payload.",
    )
    extract_parser.add_argument(
        "--wait",
        action="store_true",
        help="Poll the returned job until completion.",
    )
    extract_parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Maximum seconds to wait when --wait is used.",
    )

    poll_parser = subparsers.add_parser(
        "poll",
        help="Poll an existing job until completion.",
    )
    poll_parser.add_argument("--api-key", help="Stabilizer API key.")
    poll_parser.add_argument("--job", required=True, help="Existing job ID to poll.")
    poll_parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Maximum seconds to wait for the job to finish.",
    )

    state_parser = subparsers.add_parser(
        "state",
        help="Read ids from saved workflow state files under temp_db.",
    )
    state_parser.add_argument(
        "target",
        nargs="?",
        default="latest",
        help="'latest' or a specific run_me state file name.",
    )
    state_parser.add_argument(
        "--temp-db-dir",
        default=str(DEFAULT_TEMP_DB_DIR),
        help="Directory containing saved general and run_me state files.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _load_env_file()
    parser = _build_parser()
    args = list(argv) if argv is not None else sys.argv[1:]
    if not args:
        parser.print_help()
        return 0

    parsed = parser.parse_args(args)
    temp_db_dir = Path(getattr(parsed, "temp_db_dir", DEFAULT_TEMP_DB_DIR))

    try:
        if parsed.command == "health":
            _print_json(_make_client().health())
            return 0

        if parsed.command == "models":
            _print_json(_make_client().supported_models())
            return 0

        if parsed.command == "config":
            client = _make_client(api_key=_require_api_key(parsed.api_key))
            request_payload = LLMConfigRequest.from_payload(
                _load_config_payload(parsed.payload_file, temp_db_dir)
            ).as_payload()
            result = client.create_llm_config(request_payload)
            _save_general_payload(
                temp_db_dir,
                DEFAULT_CONFIG_FILE,
                request=request_payload,
                response=result,
            )
            _print_json(result)
            return 0

        if parsed.command == "optimize":
            client = _make_client(api_key=_require_api_key(parsed.api_key))
            request_payload = _load_request_payload(
                parsed.payload_file,
                temp_db_dir=temp_db_dir,
                default_file=DEFAULT_OPTIMIZE_FILE,
                required_keys={"prompt", "json_structure"},
                config_id=_resolve_config_id(parsed.config, temp_db_dir),
            )
            result = client.optimize_prompt(request_payload)
            if parsed.wait:
                result = client.wait_for_job(result["job_id"], timeout=parsed.timeout)
            _save_general_payload(
                temp_db_dir,
                DEFAULT_OPTIMIZE_FILE,
                request=request_payload,
                response=result,
            )
            _print_json(result)
            return 0

        if parsed.command == "compile":
            client = _make_client(api_key=_require_api_key(parsed.api_key))
            request_payload = _load_request_payload(
                parsed.payload_file,
                temp_db_dir=temp_db_dir,
                default_file=DEFAULT_COMPILE_FILE,
                required_keys={"name", "prompt", "json_structure"},
                config_id=_resolve_config_id(parsed.config, temp_db_dir),
            )
            result = client.compile_function(request_payload)
            if parsed.wait:
                result = client.wait_for_job(result["job_id"], timeout=parsed.timeout)
            _save_general_payload(
                temp_db_dir,
                DEFAULT_COMPILE_FILE,
                request=request_payload,
                response=result,
            )
            _print_json(result)
            return 0

        if parsed.command == "extract":
            client = _make_client(api_key=_require_api_key(parsed.api_key))
            request_payload = _load_request_payload(
                parsed.payload_file,
                temp_db_dir=temp_db_dir,
                default_file=DEFAULT_EXTRACT_FILE,
                required_keys={"source_text"},
                function_id=_resolve_function_id(parsed.function, temp_db_dir),
            )
            result = client.extract(request_payload)
            if parsed.wait:
                result = client.wait_for_job(result["job_id"], timeout=parsed.timeout)
            _save_general_payload(
                temp_db_dir,
                DEFAULT_EXTRACT_FILE,
                request=request_payload,
                response=result,
            )
            _print_json(result)
            return 0

        if parsed.command == "wait":
            client = _make_client(api_key=_require_api_key(parsed.api_key))
            result = client.wait_for_job(parsed.job_id, timeout=parsed.timeout)
            _print_json(result)
            return 0

        if parsed.command == "poll":
            client = _make_client(api_key=_require_api_key(parsed.api_key))
            result = _poll_job_with_progress(
                client,
                job_id=parsed.job,
                timeout=parsed.timeout,
            )
            _save_polled_job_result(temp_db_dir, result)
            _print_json(result)
            return 0

        if parsed.command == "state":
            if parsed.target == "latest":
                _print_json(_summarize_general_state(temp_db_dir))
                return 0
            _print_json(_summarize_state(_resolve_state_path(parsed.target, temp_db_dir)))
            return 0
    except (ApiError, OSError, ValueError, json.JSONDecodeError, TimeoutError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

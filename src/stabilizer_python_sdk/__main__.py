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

DEFAULT_TEMP_DB_DIR = Path("temp_db")
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


def _latest_state_file(temp_db_dir: Path) -> Path:
    state_files = sorted(path for path in temp_db_dir.glob("*.json") if path.is_file())
    if not state_files:
        raise ValueError(f"No state files found in '{temp_db_dir}'.")
    return state_files[-1]


def _resolve_state_path(target: str, temp_db_dir: Path) -> Path:
    if target == "latest":
        return _latest_state_file(temp_db_dir)
    candidate = Path(target)
    if candidate.is_file():
        return candidate
    resolved = temp_db_dir / target
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


def _list_state_summaries(temp_db_dir: Path, *, limit: int = 10) -> list[dict[str, Any]]:
    state_files = sorted(
        (path for path in temp_db_dir.glob("*.json") if path.is_file()),
        reverse=True,
    )
    return [_summarize_state(path) for path in state_files[:limit]]


def _load_request_payload(
    payload_file: str,
    *,
    config_id: str | None = None,
    function_id: str | None = None,
) -> dict[str, Any]:
    payload = _read_payload(payload_file)
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
    while True:
        job = client.get_job(job_id)
        status = str(job.get("status", "")).lower()
        progress = job.get("progress")
        if isinstance(progress, (int, float)):
            print(f"Progress: {int(progress)}% ({status or 'unknown'})")
        else:
            print(f"Progress: status={status or 'unknown'}")
        if status in TERMINAL_JOB_STATUSES:
            return job
        if time.monotonic() >= deadline:
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

    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Run POST /v1/prompt-optimizations with a JSON payload file.",
    )
    optimize_parser.add_argument("--api-key", help="Stabilizer API key.")
    optimize_parser.add_argument(
        "--payload-file",
        required=True,
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
        required=True,
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
        required=True,
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

    wait_parser = subparsers.add_parser(
        "wait",
        help="Poll an existing job ID until completion.",
    )
    wait_parser.add_argument("--api-key", help="Stabilizer API key.")
    wait_parser.add_argument("--job-id", required=True, help="Existing job ID to poll.")
    wait_parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Maximum seconds to wait for the job to finish.",
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
        help="'latest', 'list', or a specific state file name.",
    )
    state_parser.add_argument(
        "--temp-db-dir",
        default=str(DEFAULT_TEMP_DB_DIR),
        help="Directory containing saved workflow state files.",
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

    try:
        if parsed.command == "health":
            _print_json(_make_client().health())
            return 0

        if parsed.command == "models":
            _print_json(_make_client().supported_models())
            return 0

        if parsed.command == "optimize":
            client = _make_client(api_key=_require_api_key(parsed.api_key))
            result = client.optimize_prompt(
                _load_request_payload(
                    parsed.payload_file,
                    config_id=parsed.config,
                )
            )
            if parsed.wait:
                result = client.wait_for_job(result["job_id"], timeout=parsed.timeout)
            _print_json(result)
            return 0

        if parsed.command == "compile":
            client = _make_client(api_key=_require_api_key(parsed.api_key))
            result = client.compile_function(
                _load_request_payload(
                    parsed.payload_file,
                    config_id=parsed.config,
                )
            )
            if parsed.wait:
                result = client.wait_for_job(result["job_id"], timeout=parsed.timeout)
            _print_json(result)
            return 0

        if parsed.command == "extract":
            client = _make_client(api_key=_require_api_key(parsed.api_key))
            result = client.extract(
                _load_request_payload(
                    parsed.payload_file,
                    function_id=parsed.function,
                )
            )
            if parsed.wait:
                result = client.wait_for_job(result["job_id"], timeout=parsed.timeout)
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
            _print_json(result)
            return 0

        if parsed.command == "state":
            temp_db_dir = Path(parsed.temp_db_dir)
            if parsed.target == "list":
                _print_json(_list_state_summaries(temp_db_dir))
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

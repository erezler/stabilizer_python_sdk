from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from stabilizer_python_sdk import ApiError, StabilizerClient


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stabilizer_python_sdk",
        description="CLI for the Stabilizer API.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("health", help="Run GET /v1/health.")
    subparsers.add_parser("models", help="Run GET /v1/supported-models.")

    compile_parser = subparsers.add_parser(
        "compile",
        help="Run POST /v1/functions with a JSON payload file.",
    )
    compile_parser.add_argument("--api-key", required=True, help="Stabilizer API key.")
    compile_parser.add_argument(
        "--payload-file",
        required=True,
        help="Path to a JSON file for the compile request body.",
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
    extract_parser.add_argument("--api-key", required=True, help="Stabilizer API key.")
    extract_parser.add_argument(
        "--payload-file",
        required=True,
        help="Path to a JSON file for the extraction request body.",
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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
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

        if parsed.command == "compile":
            client = _make_client(api_key=parsed.api_key)
            result = client.compile_function(_read_payload(parsed.payload_file))
            if parsed.wait:
                result = client.wait_for_job(result["job_id"], timeout=parsed.timeout)
            _print_json(result)
            return 0

        if parsed.command == "extract":
            client = _make_client(api_key=parsed.api_key)
            result = client.extract(_read_payload(parsed.payload_file))
            if parsed.wait:
                result = client.wait_for_job(result["job_id"], timeout=parsed.timeout)
            _print_json(result)
            return 0
    except (ApiError, OSError, ValueError, json.JSONDecodeError, TimeoutError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

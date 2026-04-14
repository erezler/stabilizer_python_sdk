from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TextIO

STATE_FILE_TIMESTAMP_FORMAT = "%Y-%m-%d-%H-%M-%S"
TERMINAL_JOB_STATUSES = frozenset({"completed", "failed"})
LatestStatePathResolver = Callable[[], datetime]


class WorkflowConsole:
    _RESET = "\x1b[0m"
    _BOLD = "\x1b[1m"
    _BLUE = "\x1b[34m"
    _CYAN = "\x1b[36m"
    _GREEN = "\x1b[32m"
    _YELLOW = "\x1b[33m"
    _RED = "\x1b[31m"

    def __init__(self, *, stream: TextIO | None = None, use_color: bool = True) -> None:
        self._stream = stream or sys.stdout
        self._use_color = use_color

    def section(self, step_name: str, message: str) -> None:
        self._emit(step_name, message, self._BLUE)

    def info(self, step_name: str, message: str) -> None:
        self._emit(step_name, message, self._CYAN)

    def progress(self, step_name: str, message: str) -> None:
        self._emit(step_name, message, self._CYAN)

    def success(self, step_name: str, message: str) -> None:
        self._emit(step_name, message, self._GREEN)

    def skip(self, step_name: str, message: str) -> None:
        self._emit(step_name, message, self._YELLOW)

    def error(self, step_name: str, message: str) -> None:
        self._emit(step_name, message, self._RED)

    def _emit(self, step_name: str, message: str, color: str) -> None:
        prefix = f"[{step_name.upper()}]"
        if self._use_color:
            line = f"{self._BOLD}{color}{prefix}{self._RESET} {message}"
        else:
            line = f"{prefix} {message}"
        print(line, file=self._stream)
        self._stream.flush()


def default_now_provider() -> datetime:
    return datetime.now()


def timestamp_string(now_provider: LatestStatePathResolver = default_now_provider) -> str:
    return now_provider().strftime(STATE_FILE_TIMESTAMP_FORMAT)


def resolve_state_file(
    *,
    state_file: str | Path | None,
    temp_db_dir: str | Path,
    force_new: bool = False,
    now_provider: LatestStatePathResolver = default_now_provider,
) -> Path:
    if state_file is not None:
        return Path(state_file)

    temp_db_path = Path(temp_db_dir)
    temp_db_path.mkdir(parents=True, exist_ok=True)
    if force_new:
        return temp_db_path / f"{timestamp_string(now_provider)}.json"
    existing = sorted(path for path in temp_db_path.glob("*.json") if path.is_file())
    if existing:
        return existing[-1]
    return temp_db_path / f"{timestamp_string(now_provider)}.json"


def load_state(
    *,
    state_file: str | Path | None,
    temp_db_dir: str | Path,
    now_provider: LatestStatePathResolver = default_now_provider,
) -> tuple[Path, dict[str, Any]]:
    path = resolve_state_file(
        state_file=state_file,
        temp_db_dir=temp_db_dir,
        now_provider=now_provider,
    )
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    else:
        current_time = timestamp_string(now_provider)
        state = {
            "created_at": current_time,
            "updated_at": current_time,
            "steps": {},
        }
    state.setdefault("steps", {})
    return path, state


def save_state(
    path: str | Path,
    state: dict[str, Any],
    *,
    now_provider: LatestStatePathResolver = default_now_provider,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state.setdefault("created_at", timestamp_string(now_provider))
    state["updated_at"] = timestamp_string(now_provider)
    target.write_text(json.dumps(state, indent=2), encoding="utf-8")


def is_step_complete(state: dict[str, Any], step_name: str) -> bool:
    step_data = state.get("steps", {}).get(step_name)
    return isinstance(step_data, dict) and "result" in step_data


def get_step_result(state: dict[str, Any], step_name: str) -> dict[str, Any] | None:
    step_data = state.get("steps", {}).get(step_name)
    if isinstance(step_data, dict):
        result = step_data.get("result")
        if isinstance(result, dict):
            return result
    return None


def get_saved_request(state: dict[str, Any], step_name: str) -> dict[str, Any] | None:
    step_data = state.get("steps", {}).get(step_name)
    if isinstance(step_data, dict):
        request = step_data.get("request")
        if isinstance(request, dict):
            return request
    return None


def update_step_state(
    state: dict[str, Any],
    *,
    step_name: str,
    request: dict[str, Any],
    result: dict[str, Any],
    submission: dict[str, Any] | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    step_state: dict[str, Any] = {
        "request": request,
        "result": result,
    }
    if submission is not None:
        step_state["submission"] = submission
    if extra_fields:
        step_state.update(extra_fields)
    state.setdefault("steps", {})[step_name] = step_state
    return step_state


def load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in '{path}'.")
    return payload


def poll_job(
    client: Any,
    *,
    job_id: str,
    step_name: str,
    console: WorkflowConsole,
    poll_interval: float,
    timeout: float,
    sleeper: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    deadline = monotonic() + timeout
    while True:
        job = client.get_job(job_id)
        status = str(job.get("status", "")).lower()
        progress = job.get("progress")
        if isinstance(progress, (int, float)):
            console.progress(step_name, f"Polling job {job_id}: {int(progress)}% ({status or 'unknown'})")
        else:
            console.progress(step_name, f"Polling job {job_id}: status={status or 'unknown'}")
        if status in TERMINAL_JOB_STATUSES:
            return job
        if monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for job '{job_id}'.")
        sleeper(poll_interval)

from __future__ import annotations

import json
from pathlib import Path

import pytest

import stabilizer_python_sdk.__main__ as cli


class FakeClient:
    def __init__(self, *, api_key: str | None = None) -> None:
        self.api_key = api_key
        self.calls: list[tuple[str, object]] = []

    def health(self) -> dict[str, object]:
        self.calls.append(("health", None))
        return {"status": "ok", "version": "v1"}

    def supported_models(self) -> dict[str, object]:
        self.calls.append(("supported_models", None))
        return {"models": ["google/gemini-2.5-flash-lite"]}

    def compile_function(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("compile_function", payload))
        return {"job_id": "job_compile", "status": "queued"}

    def optimize_prompt(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("optimize_prompt", payload))
        return {"job_id": "job_optimize", "status": "queued"}

    def extract(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("extract", payload))
        return {"job_id": "job_extract", "status": "queued"}

    def wait_for_job(self, job_id: str, *, timeout: float) -> dict[str, object]:
        self.calls.append(("wait_for_job", {"job_id": job_id, "timeout": timeout}))
        return {"job_id": job_id, "status": "completed", "result": {"ok": True}}


def test_main_without_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main([])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "usage:" in captured.out
    assert "health" in captured.out
    assert "models" in captured.out
    assert "compile" in captured.out
    assert "extract" in captured.out


def _write_state(
    temp_db_dir: Path,
    filename: str,
    *,
    config_id: str | None = None,
    optimize_job_id: str | None = None,
    compile_job_id: str | None = None,
    function_id: str | None = None,
    extract_job_id: str | None = None,
) -> None:
    state = {
        "created_at": filename.removesuffix(".json"),
        "updated_at": filename.removesuffix(".json"),
        "steps": {},
    }
    if config_id is not None:
        state["steps"]["config"] = {
            "config_id": config_id,
            "result": {"config_id": config_id},
            "request": {"name": "Primary config"},
        }
    if optimize_job_id is not None:
        state["steps"]["optimize"] = {
            "job_id": optimize_job_id,
            "result": {"job_id": optimize_job_id},
            "request": {"prompt": "Optimize"},
        }
    if compile_job_id is not None or function_id is not None:
        state["steps"]["compile"] = {
            "job_id": compile_job_id,
            "function_id": function_id,
            "result": {"job_id": compile_job_id, "result": {"function_id": function_id}},
            "request": {"name": "Compile"},
        }
    if extract_job_id is not None:
        state["steps"]["extract"] = {
            "job_id": extract_job_id,
            "result": {"job_id": extract_job_id},
            "request": {"source_text": "hello"},
        }
    (temp_db_dir / filename).write_text(json.dumps(state), encoding="utf-8")


def test_health_command_prints_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_client = FakeClient()
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(["health"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == {"status": "ok", "version": "v1"}
    assert fake_client.calls == [("health", None)]


def test_models_command_prints_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_client = FakeClient()
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(["models"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == {"models": ["google/gemini-2.5-flash-lite"]}
    assert fake_client.calls == [("supported_models", None)]


def test_compile_command_loads_payload_file_and_can_wait_for_result(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {"name": "Example", "prompt": "Extract", "json_structure": {"field": "string"}}
    payload_path = tmp_path / "compile.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    fake_client = FakeClient(api_key="sk_test")
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(
        [
            "compile",
            "--api-key",
            "sk_test",
            "--payload-file",
            str(payload_path),
            "--wait",
            "--timeout",
            "42",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == {
        "job_id": "job_compile",
        "status": "completed",
        "result": {"ok": True},
    }
    assert fake_client.calls == [
        ("compile_function", payload),
        ("wait_for_job", {"job_id": "job_compile", "timeout": 42.0}),
    ]


def test_optimize_command_loads_payload_file_and_injects_config_id(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {"prompt": "Extract", "json_structure": {"field": "string"}, "training_data": []}
    payload_path = tmp_path / "optimize.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    fake_client = FakeClient(api_key="sk_test")
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(
        [
            "optimize",
            "--api-key",
            "sk_test",
            "--payload-file",
            str(payload_path),
            "--config",
            "cfg_123",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == {"job_id": "job_optimize", "status": "queued"}
    assert fake_client.calls == [
        (
            "optimize_prompt",
            {
                "prompt": "Extract",
                "json_structure": {"field": "string"},
                "training_data": [],
                "config_id": "cfg_123",
            },
        )
    ]


def test_compile_command_injects_config_id(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {"name": "Example", "prompt": "Extract", "json_structure": {"field": "string"}}
    payload_path = tmp_path / "compile.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    fake_client = FakeClient(api_key="sk_test")
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(
        [
            "compile",
            "--api-key",
            "sk_test",
            "--payload-file",
            str(payload_path),
            "--config",
            "cfg_123",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == {"job_id": "job_compile", "status": "queued"}
    assert fake_client.calls == [
        (
            "compile_function",
            {
                "name": "Example",
                "prompt": "Extract",
                "json_structure": {"field": "string"},
                "config_id": "cfg_123",
            },
        )
    ]


def test_extract_command_loads_payload_file_and_prints_queued_job(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {"function_id": "fn_123", "source_text": "hello"}
    payload_path = tmp_path / "extract.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    fake_client = FakeClient(api_key="sk_test")
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(
        [
            "extract",
            "--api-key",
            "sk_test",
            "--payload-file",
            str(payload_path),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == {"job_id": "job_extract", "status": "queued"}
    assert fake_client.calls == [("extract", payload)]


def test_extract_command_injects_function_id(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {"function_id": "fn_replace_me", "source_text": "hello"}
    payload_path = tmp_path / "extract.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    fake_client = FakeClient(api_key="sk_test")
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(
        [
            "extract",
            "--api-key",
            "sk_test",
            "--payload-file",
            str(payload_path),
            "--function",
            "fn_123",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == {"job_id": "job_extract", "status": "queued"}
    assert fake_client.calls == [
        (
            "extract",
            {
                "function_id": "fn_123",
                "source_text": "hello",
            },
        )
    ]


def test_state_latest_prints_latest_state_ids(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    temp_db_dir = tmp_path / "temp_db"
    temp_db_dir.mkdir()
    _write_state(
        temp_db_dir,
        "2026-04-14-10-19-17.json",
        config_id="cfg_old",
        optimize_job_id="job_opt_old",
        compile_job_id="job_compile_old",
        function_id="fn_old",
        extract_job_id="job_extract_old",
    )
    _write_state(
        temp_db_dir,
        "2026-04-14-10-19-18.json",
        config_id="cfg_123",
        optimize_job_id="job_opt_123",
        compile_job_id="job_compile_123",
        function_id="fn_123",
        extract_job_id="job_extract_123",
    )
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(["state", "latest"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == {
        "state_file": "2026-04-14-10-19-18.json",
        "config_id": "cfg_123",
        "optimize_job_id": "job_opt_123",
        "compile_job_id": "job_compile_123",
        "function_id": "fn_123",
        "extract_job_id": "job_extract_123",
    }


def test_state_list_prints_last_ten_states(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    temp_db_dir = tmp_path / "temp_db"
    temp_db_dir.mkdir()
    for index in range(12):
        _write_state(
            temp_db_dir,
            f"2026-04-14-10-19-{index:02d}.json",
            config_id=f"cfg_{index:02d}",
            optimize_job_id=f"job_opt_{index:02d}",
            compile_job_id=f"job_compile_{index:02d}",
            function_id=f"fn_{index:02d}",
            extract_job_id=f"job_extract_{index:02d}",
        )
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(["state", "list"])

    captured = capsys.readouterr()
    items = json.loads(captured.out)

    assert exit_code == 0
    assert len(items) == 10
    assert items[0]["state_file"] == "2026-04-14-10-19-11.json"
    assert items[-1]["state_file"] == "2026-04-14-10-19-02.json"


def test_state_file_name_prints_specific_state_ids(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    temp_db_dir = tmp_path / "temp_db"
    temp_db_dir.mkdir()
    _write_state(
        temp_db_dir,
        "2026-04-14-10-19-18.json",
        config_id="cfg_123",
        optimize_job_id="job_opt_123",
        compile_job_id="job_compile_123",
        function_id="fn_123",
        extract_job_id="job_extract_123",
    )
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(["state", "2026-04-14-10-19-18.json"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == {
        "state_file": "2026-04-14-10-19-18.json",
        "config_id": "cfg_123",
        "optimize_job_id": "job_opt_123",
        "compile_job_id": "job_compile_123",
        "function_id": "fn_123",
        "extract_job_id": "job_extract_123",
    }


def test_wait_command_polls_existing_job_id(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_client = FakeClient(api_key="sk_test")
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(
        [
            "wait",
            "--api-key",
            "sk_test",
            "--job-id",
            "job_123",
            "--timeout",
            "90",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == {
        "job_id": "job_123",
        "status": "completed",
        "result": {"ok": True},
    }
    assert fake_client.calls == [
        ("wait_for_job", {"job_id": "job_123", "timeout": 90.0}),
    ]

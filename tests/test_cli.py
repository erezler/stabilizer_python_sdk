from __future__ import annotations

import json

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
        return {"models": ["openai/gpt-5.4-mini"]}

    def compile_function(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("compile_function", payload))
        return {"job_id": "job_compile", "status": "queued"}

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
    assert json.loads(captured.out) == {"models": ["openai/gpt-5.4-mini"]}
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

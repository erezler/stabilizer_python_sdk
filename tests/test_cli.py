from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

import pytest

import stabilizer_python_sdk.__main__ as cli


class FakeClient:
    def __init__(self, *, api_key: str | None = None) -> None:
        self.api_key = api_key
        self.calls: list[tuple[str, object]] = []
        self._jobs: dict[str, list[dict[str, object]]] = {}

    def health(self) -> dict[str, object]:
        self.calls.append(("health", None))
        return {"status": "ok", "version": "v1"}

    def supported_models(self) -> dict[str, object]:
        self.calls.append(("supported_models", None))
        return {"models": ["google/gemini-2.5-flash-lite"]}

    def get_org(self) -> dict[str, object]:
        self.calls.append(("get_org", None))
        return {"org_id": "org_123", "name": "Acme"}

    def update_org(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("update_org", payload))
        return {"org_id": "org_123", **payload}

    def list_api_keys(self) -> list[dict[str, object]]:
        self.calls.append(("list_api_keys", None))
        return [{"key_id": "key_123", "name": "Primary", "scope": "full", "revoked": False}]

    def create_api_key(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("create_api_key", payload))
        return {
            "key_id": "key_456",
            "name": str(payload.get("name", "New key")),
            "scope": str(payload.get("scope", "full")),
            "key_value": "sk_live_123",
        }

    def get_api_key(self, key_id: str) -> dict[str, object]:
        self.calls.append(("get_api_key", {"key_id": key_id}))
        return {"key_id": key_id, "name": "Primary", "scope": "full", "revoked": False}

    def update_api_key(self, key_id: str, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("update_api_key", {"key_id": key_id, "payload": payload}))
        return {"key_id": key_id, **payload}

    def get_api_key_usage(
        self,
        key_id: str,
        *,
        from_: str | None = None,
        to: str | None = None,
    ) -> dict[str, object]:
        self.calls.append(("get_api_key_usage", {"key_id": key_id, "from": from_, "to": to}))
        return {"key_id": key_id, "from": from_, "to": to, "extract_count": 4}

    def revoke_api_key(self, key_id: str) -> None:
        self.calls.append(("revoke_api_key", {"key_id": key_id}))
        return None

    def list_llm_configs(self) -> list[dict[str, object]]:
        self.calls.append(("list_llm_configs", None))
        return [{"config_id": "cfg_123", "name": "Primary config"}]

    def create_llm_config(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("create_llm_config", payload))
        return {"config_id": "cfg_123", "status": "created"}

    def update_llm_config(self, config_id: str, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("update_llm_config", {"config_id": config_id, "payload": payload}))
        return {"config_id": config_id, **payload}

    def delete_llm_config(self, config_id: str) -> None:
        self.calls.append(("delete_llm_config", {"config_id": config_id}))
        return None

    def test_llm_config(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("test_llm_config", payload))
        return {"valid": True, "reason": "ok"}

    def list_functions(self, *, name: str | None = None, tag: str | None = None) -> list[dict[str, object]]:
        self.calls.append(("list_functions", {"name": name, "tag": tag}))
        return [{"function_id": "fn_123", "name": name or "Invoice extractor", "tags": [tag] if tag else []}]

    def get_function(self, function_id: str) -> dict[str, object]:
        self.calls.append(("get_function", {"function_id": function_id}))
        return {"function_id": function_id, "name": "Invoice extractor"}

    def update_function(self, function_id: str, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("update_function", {"function_id": function_id, "payload": payload}))
        return {"function_id": function_id, **payload}

    def delete_function(self, function_id: str) -> None:
        self.calls.append(("delete_function", {"function_id": function_id}))
        return None

    def compile_function(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("compile_function", payload))
        return {"job_id": "job_compile", "status": "queued"}

    def optimize_prompt(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("optimize_prompt", payload))
        return {"job_id": "job_optimize", "status": "queued"}

    def extract(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("extract", payload))
        return {"job_id": "job_extract", "status": "queued"}

    def list_extractions(self) -> list[dict[str, object]]:
        self.calls.append(("list_extractions", None))
        return [{"job_id": "job_extract", "type": "extract", "status": "completed"}]

    def set_job_sequence(self, job_id: str, jobs: list[dict[str, object]]) -> None:
        self._jobs[job_id] = list(jobs)

    def get_job(self, job_id: str) -> dict[str, object]:
        self.calls.append(("get_job", {"job_id": job_id}))
        sequence = self._jobs[job_id]
        if len(sequence) > 1:
            return sequence.pop(0)
        return sequence[0]

    def get_usage(
        self,
        *,
        from_: str | None = None,
        to: str | None = None,
        group_by: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            (
                "get_usage",
                {"from": from_, "to": to, "group_by": group_by, "limit": limit, "cursor": cursor},
            )
        )
        return {"org_id": "org_123", "from": from_, "to": to, "total_jobs": 7}

    def evaluate_variance(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("evaluate_variance", payload))
        return {"score": 0.91}

    def evaluate_ground_truth(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(("evaluate_ground_truth", payload))
        return {"score": 0.98}


def test_main_without_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main([])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "usage:" in captured.out
    assert "health" in captured.out
    assert "models" in captured.out
    assert "config" in captured.out
    assert "compile" in captured.out
    assert "extract" in captured.out
    assert "org" in captured.out
    assert "api-keys" in captured.out
    assert "configs" in captured.out
    assert "functions" in captured.out
    assert "usage" in captured.out
    assert "evaluate-variance" not in captured.out
    assert "evaluate-gt" not in captured.out
    assert "wait" not in captured.out


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


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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


def test_org_command_prints_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_client = FakeClient()
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(["org", "--api-key", "sk_test"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == {"org_id": "org_123", "name": "Acme"}
    assert fake_client.calls == [("get_org", None)]


def test_org_update_command_loads_payload_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload_path = tmp_path / "org-update.json"
    _write_json(payload_path, {"name": "Renamed org"})
    fake_client = FakeClient()
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(["org-update", "--api-key", "sk_test", "--payload-file", str(payload_path)])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == {"org_id": "org_123", "name": "Renamed org"}
    assert fake_client.calls == [("update_org", {"name": "Renamed org"})]


def test_api_keys_command_prints_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_client = FakeClient()
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(["api-keys", "--api-key", "sk_test"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == [{"key_id": "key_123", "name": "Primary", "scope": "full", "revoked": False}]
    assert fake_client.calls == [("list_api_keys", None)]


def test_api_key_create_command_loads_payload_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload_path = tmp_path / "api-key-create.json"
    _write_json(payload_path, {"name": "CLI key", "scope": "read_only"})
    fake_client = FakeClient()
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(["api-key-create", "--api-key", "sk_test", "--payload-file", str(payload_path)])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == {
        "key_id": "key_456",
        "name": "CLI key",
        "scope": "read_only",
        "key_value": "sk_live_123",
    }
    assert fake_client.calls == [("create_api_key", {"name": "CLI key", "scope": "read_only"})]


def test_api_key_revoke_command_calls_client(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_client = FakeClient()
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(["api-key-revoke", "--api-key", "sk_test", "--key", "key_123"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) is None
    assert fake_client.calls == [("revoke_api_key", {"key_id": "key_123"})]


def test_api_key_command_fetches_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_client = FakeClient()
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(["api-key", "--api-key", "sk_test", "--key", "me"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == {"key_id": "me", "name": "Primary", "scope": "full", "revoked": False}
    assert fake_client.calls == [("get_api_key", {"key_id": "me"})]


def test_api_key_update_command_loads_payload_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload_path = tmp_path / "api-key-update.json"
    _write_json(payload_path, {"budget": {"extract_limit": 0}, "revoked": True})
    fake_client = FakeClient()
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(
        ["api-key-update", "--api-key", "sk_test", "--key", "key_123", "--payload-file", str(payload_path)]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == {"key_id": "key_123", "budget": {"extract_limit": 0}, "revoked": True}
    assert fake_client.calls == [
        ("update_api_key", {"key_id": "key_123", "payload": {"budget": {"extract_limit": 0}, "revoked": True}})
    ]


def test_api_key_usage_command_forwards_query_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_client = FakeClient()
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(
        ["api-key-usage", "--api-key", "sk_test", "--key", "key_123", "--from", "2026-04-01", "--to", "2026-04-15"]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == {
        "key_id": "key_123",
        "from": "2026-04-01",
        "to": "2026-04-15",
        "extract_count": 4,
    }
    assert fake_client.calls == [
        ("get_api_key_usage", {"key_id": "key_123", "from": "2026-04-01", "to": "2026-04-15"})
    ]


def test_configs_command_prints_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_client = FakeClient()
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(["configs", "--api-key", "sk_test"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == [{"config_id": "cfg_123", "name": "Primary config"}]
    assert fake_client.calls == [("list_llm_configs", None)]


def test_config_update_command_loads_payload_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload_path = tmp_path / "config-update.json"
    _write_json(payload_path, {"name": "Renamed config"})
    fake_client = FakeClient()
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(
        ["config-update", "--api-key", "sk_test", "--config", "cfg_123", "--payload-file", str(payload_path)]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == {"config_id": "cfg_123", "name": "Renamed config"}
    assert fake_client.calls == [("update_llm_config", {"config_id": "cfg_123", "payload": {"name": "Renamed config"}})]


def test_config_delete_command_calls_client(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_client = FakeClient()
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(["config-delete", "--api-key", "sk_test", "--config", "cfg_123"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) is None
    assert fake_client.calls == [("delete_llm_config", {"config_id": "cfg_123"})]


def test_config_test_command_verifies_provider_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload_path = tmp_path / "config-test.json"
    _write_json(payload_path, {"api_key": "provider-key", "default_model": "google/gemini-2.5-flash-lite"})
    monkeypatch.delenv("STABILIZER_PROVIDER_API_KEY", raising=False)
    fake_client = FakeClient()
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(["config-test", "--api-key", "sk_test", "--payload-file", str(payload_path)])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == {"valid": True, "reason": "ok"}
    assert fake_client.calls == [
        ("test_llm_config", {"api_key": "provider-key", "default_model": "google/gemini-2.5-flash-lite"})
    ]


def test_functions_command_forwards_filters(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_client = FakeClient()
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(["functions", "--api-key", "sk_test", "--name", "Invoice", "--tag", "billing"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == [{"function_id": "fn_123", "name": "Invoice", "tags": ["billing"]}]
    assert fake_client.calls == [("list_functions", {"name": "Invoice", "tag": "billing"})]


def test_function_command_prints_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_client = FakeClient()
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(["function", "--api-key", "sk_test", "--function", "fn_123"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == {"function_id": "fn_123", "name": "Invoice extractor"}
    assert fake_client.calls == [("get_function", {"function_id": "fn_123"})]


def test_function_update_command_loads_payload_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload_path = tmp_path / "function-update.json"
    _write_json(payload_path, {"name": "Renamed function"})
    fake_client = FakeClient()
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(
        ["function-update", "--api-key", "sk_test", "--function", "fn_123", "--payload-file", str(payload_path)]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == {"function_id": "fn_123", "name": "Renamed function"}
    assert fake_client.calls == [
        ("update_function", {"function_id": "fn_123", "payload": {"name": "Renamed function"}})
    ]


def test_function_delete_command_calls_client(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_client = FakeClient()
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(["function-delete", "--api-key", "sk_test", "--function", "fn_123"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) is None
    assert fake_client.calls == [("delete_function", {"function_id": "fn_123"})]


def test_job_command_fetches_status_without_polling(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_client = FakeClient()
    fake_client.set_job_sequence("job_123", [{"job_id": "job_123", "status": "running", "progress": 40}])
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(["job", "--api-key", "sk_test", "--job", "job_123"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == {"job_id": "job_123", "status": "running", "progress": 40}
    assert fake_client.calls == [("get_job", {"job_id": "job_123"})]


def test_extractions_command_prints_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_client = FakeClient()
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(["extractions", "--api-key", "sk_test"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == [{"job_id": "job_extract", "type": "extract", "status": "completed"}]
    assert fake_client.calls == [("list_extractions", None)]


def test_usage_command_forwards_query_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_client = FakeClient()
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(["usage", "--api-key", "sk_test", "--from", "2026-04-01", "--to", "2026-04-15"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == {
        "org_id": "org_123",
        "from": "2026-04-01",
        "to": "2026-04-15",
        "total_jobs": 7,
    }
    assert fake_client.calls == [
        (
            "get_usage",
            {"from": "2026-04-01", "to": "2026-04-15", "group_by": None, "limit": None, "cursor": None},
        )
    ]


def test_usage_command_forwards_group_by_limit_and_cursor(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_client = FakeClient()
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(
        [
            "usage",
            "--api-key",
            "sk_test",
            "--from",
            "2026-04-01",
            "--to",
            "2026-04-15",
            "--group-by",
            "key",
            "--limit",
            "50",
            "--cursor",
            "key_123",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert fake_client.calls == [
        (
            "get_usage",
            {
                "from": "2026-04-01",
                "to": "2026-04-15",
                "group_by": "key",
                "limit": 50,
                "cursor": "key_123",
            },
        )
    ]


@pytest.mark.parametrize("command", ["evaluate-variance", "evaluate-gt"])
def test_removed_evaluation_commands_are_not_available(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main([command])

    captured = capsys.readouterr()

    assert exc_info.value.code == 2
    assert "invalid choice" in captured.err
    assert command in captured.err


def test_config_command_requires_payload_file(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / ".env.local").write_text("STABILIZER_API_KEY=sk_from_env\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("STABILIZER_API_KEY", raising=False)
    monkeypatch.delenv("STABILIZER_PROVIDER_API_KEY", raising=False)
    reloaded = importlib.reload(cli)

    exit_code = reloaded.main(["config"])

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "--payload-file" in captured.err


def test_config_command_uses_provider_api_key_from_env_when_payload_omits_it(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {
        "name": "Primary config",
        "provider": "openai",
        "default_model": "google/gemini-2.5-flash-lite",
        "is_default": True,
        "byok": True,
    }
    payload_path = tmp_path / "config-input.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / ".env.local").write_text(
        "STABILIZER_API_KEY=sk_from_env\nSTABILIZER_PROVIDER_API_KEY=provider_from_env\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("STABILIZER_API_KEY", raising=False)
    monkeypatch.delenv("STABILIZER_PROVIDER_API_KEY", raising=False)
    reloaded = importlib.reload(cli)
    fake_client = FakeClient(api_key="sk_from_env")
    captured_api_keys: list[str | None] = []

    def fake_make_client(api_key=None):
        captured_api_keys.append(api_key)
        return fake_client

    monkeypatch.setattr(reloaded, "_make_client", fake_make_client)

    exit_code = reloaded.main(["config", "--payload-file", str(payload_path)])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == {"config_id": "cfg_123", "status": "created"}
    assert captured_api_keys == ["sk_from_env"]
    assert fake_client.calls == [
        (
            "create_llm_config",
            {
                "name": "Primary config",
                "provider": "openai",
                "default_model": "google/gemini-2.5-flash-lite",
                "is_default": True,
                "byok": True,
                "api_key": "provider_from_env",
            },
        )
    ]


def test_compile_command_uses_api_key_from_env_local(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {"name": "Example", "prompt": "Extract", "json_structure": {"field": "string"}}
    payload_path = tmp_path / "compile-input.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / ".env.local").write_text("STABILIZER_API_KEY=sk_from_env\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("STABILIZER_API_KEY", raising=False)
    reloaded = importlib.reload(cli)
    fake_client = FakeClient(api_key="sk_from_env")
    captured_api_keys: list[str | None] = []

    def fake_make_client(api_key=None):
        captured_api_keys.append(api_key)
        return fake_client

    monkeypatch.setattr(reloaded, "_make_client", fake_make_client)

    exit_code = reloaded.main(
        [
            "compile",
            "--payload-file",
            str(payload_path),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == {"job_id": "job_compile", "status": "queued"}
    assert captured_api_keys == ["sk_from_env"]
    assert fake_client.calls == [("compile_function", payload)]


def test_compile_command_errors_when_api_key_missing_everywhere(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {"name": "Example", "prompt": "Extract", "json_structure": {"field": "string"}}
    payload_path = tmp_path / "compile-input.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("STABILIZER_API_KEY", raising=False)
    reloaded = importlib.reload(cli)

    exit_code = reloaded.main(
        [
            "compile",
            "--payload-file",
            str(payload_path),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "STABILIZER_API_KEY" in captured.err


def test_compile_command_requires_payload_file(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "compile-input.json").write_text(
        json.dumps({"name": "Example", "prompt": "Extract", "json_structure": {"field": "string"}}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(["compile", "--api-key", "sk_test"])

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "--payload-file" in captured.err


def test_compile_command_loads_payload_file_and_can_poll_for_result(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {"name": "Example", "prompt": "Extract", "json_structure": {"field": "string"}}
    payload_path = tmp_path / "compile-input.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    fake_client = FakeClient(api_key="sk_test")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    fake_client.set_job_sequence(
        "job_compile",
        [
            {"job_id": "job_compile", "status": "running", "progress": 40},
            {"job_id": "job_compile", "status": "completed", "progress": 100, "result": {"ok": True}},
        ],
    )

    exit_code = cli.main(
        [
            "compile",
            "--api-key",
            "sk_test",
            "--payload-file",
            str(payload_path),
            "--poll",
            "--timeout",
            "42",
        ]
    )

    captured = capsys.readouterr()
    saved = json.loads((tmp_path / "temp_db" / "general" / "compile-output.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert "\rProgress: 40% (running)" in captured.out
    assert "\rProgress: 100% (completed)" in captured.out
    assert json.loads(captured.out.split("\n", 1)[1]) == {
        "job_id": "job_compile",
        "status": "completed",
        "progress": 100,
        "result": {"ok": True},
    }
    assert fake_client.calls == [
        ("compile_function", payload),
        ("get_job", {"job_id": "job_compile"}),
        ("get_job", {"job_id": "job_compile"}),
    ]
    assert saved == {
        "job_id": "job_compile",
        "status": "completed",
        "progress": 100,
        "result": {"ok": True},
    }


def test_wait_command_is_not_supported(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["wait", "--api-key", "sk_test", "--job", "job_123"])

    captured = capsys.readouterr()

    assert exc_info.value.code == 2
    assert "invalid choice: 'wait'" in captured.err


def test_optimize_command_loads_payload_file_and_injects_config_id(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {"prompt": "Extract", "json_structure": {"field": "string"}, "training_data": []}
    payload_path = tmp_path / "optimize-input.json"
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


def test_optimize_command_with_poll_and_alter_compile_updates_compile_payload_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    optimize_payload = {"prompt": "Extract", "json_structure": {"field": "string"}, "training_data": []}
    optimize_payload_path = tmp_path / "optimize-input.json"
    compile_payload_path = tmp_path / "compile-input.json"
    _write_json(optimize_payload_path, optimize_payload)
    _write_json(
        compile_payload_path,
        {
            "name": "Example",
            "prompt": "Old prompt",
            "json_structure": {"field": "string"},
            "training_data": [],
        },
    )
    fake_client = FakeClient(api_key="sk_test")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    fake_client.set_job_sequence(
        "job_optimize",
        [
            {"job_id": "job_optimize", "status": "running", "progress": 40},
            {
                "job_id": "job_optimize",
                "status": "completed",
                "progress": 100,
                "result": {"optimized_prompt": "Better prompt"},
            },
        ],
    )

    exit_code = cli.main(
        [
            "optimize",
            "--api-key",
            "sk_test",
            "--payload-file",
            str(optimize_payload_path),
            "--poll",
            "--alter-compile",
            str(compile_payload_path),
        ]
    )

    captured = capsys.readouterr()
    saved_compile_payload = json.loads(compile_payload_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert json.loads(captured.out.split("\n", 1)[1]) == {
        "job_id": "job_optimize",
        "status": "completed",
        "progress": 100,
        "result": {"optimized_prompt": "Better prompt"},
    }
    assert saved_compile_payload == {
        "name": "Example",
        "prompt": "Old prompt",
        "json_structure": {"field": "string"},
        "training_data": [],
        "compile_options": {
            "optimized_prompts": ["Better prompt"],
        },
    }


def test_optimize_command_with_alter_compile_uses_first_optimized_prompt_from_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    optimize_payload = {"prompt": "Extract", "json_structure": {"field": "string"}, "training_data": []}
    optimize_payload_path = tmp_path / "optimize-input.json"
    compile_payload_path = tmp_path / "compile-input.json"
    _write_json(optimize_payload_path, optimize_payload)
    _write_json(
        compile_payload_path,
        {
            "name": "Example",
            "prompt": "Old prompt",
            "json_structure": {"field": "string"},
            "training_data": [],
        },
    )
    fake_client = FakeClient(api_key="sk_test")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    fake_client.set_job_sequence(
        "job_optimize",
        [
            {"job_id": "job_optimize", "status": "running", "progress": 40},
            {
                "job_id": "job_optimize",
                "status": "completed",
                "progress": 100,
                "result": {
                    "optimized_prompts": [
                        "Better prompt 1",
                        "Better prompt 2",
                    ]
                },
            },
        ],
    )

    exit_code = cli.main(
        [
            "optimize",
            "--api-key",
            "sk_test",
            "--payload-file",
            str(optimize_payload_path),
            "--poll",
            "--alter-compile",
            str(compile_payload_path),
        ]
    )

    captured = capsys.readouterr()
    saved_compile_payload = json.loads(compile_payload_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert json.loads(captured.out.split("\n", 1)[1]) == {
        "job_id": "job_optimize",
        "status": "completed",
        "progress": 100,
        "result": {"optimized_prompts": ["Better prompt 1", "Better prompt 2"]},
    }
    assert saved_compile_payload == {
        "name": "Example",
        "prompt": "Old prompt",
        "json_structure": {"field": "string"},
        "training_data": [],
        "compile_options": {
            "optimized_prompts": ["Better prompt 1", "Better prompt 2"],
        },
    }


def test_optimize_command_rejects_alter_compile_without_poll(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    optimize_payload_path = tmp_path / "optimize-input.json"
    compile_payload_path = tmp_path / "compile-input.json"
    _write_json(optimize_payload_path, {"prompt": "Extract", "json_structure": {"field": "string"}, "training_data": []})
    _write_json(
        compile_payload_path,
        {"name": "Example", "prompt": "Old prompt", "json_structure": {"field": "string"}},
    )
    fake_client = FakeClient(api_key="sk_test")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(
        [
            "optimize",
            "--api-key",
            "sk_test",
            "--payload-file",
            str(optimize_payload_path),
            "--alter-compile",
            str(compile_payload_path),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "--alter-compile requires --poll" in captured.err


def test_compile_command_injects_config_id(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {"name": "Example", "prompt": "Extract", "json_structure": {"field": "string"}}
    payload_path = tmp_path / "compile-input.json"
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
    payload_path = tmp_path / "extract-input.json"
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
    payload_path = tmp_path / "extract-input.json"
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
    _write_json(
        tmp_path / "temp_db" / "general" / "config-output.json",
        {
            "config_id": "cfg_123",
        },
    )
    _write_json(
        tmp_path / "temp_db" / "general" / "optimize-output.json",
        {
            "job_id": "job_opt_123",
        },
    )
    _write_json(
        tmp_path / "temp_db" / "general" / "compile-output.json",
        {
            "job_id": "job_compile_123",
            "result": {"function_id": "fn_123"},
        },
    )
    _write_json(
        tmp_path / "temp_db" / "general" / "extract-output.json",
        {
            "job_id": "job_extract_123",
        },
    )
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(["state", "latest"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == {
        "state_file": "general",
        "config_id": "cfg_123",
        "optimize_job_id": "job_opt_123",
        "compile_job_id": "job_compile_123",
        "function_id": "fn_123",
        "extract_job_id": "job_extract_123",
    }


def test_state_list_is_not_supported(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(["state", "list"])

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "State file 'list' was not found." in captured.err


def test_state_file_name_prints_specific_state_ids(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    temp_db_dir = tmp_path / "temp_db" / "run_me"
    temp_db_dir.mkdir(parents=True)
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


def test_poll_command_polls_existing_job_id(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_client = FakeClient(api_key="sk_test")
    fake_client.set_job_sequence(
        "job_75b0bf3c5d5541f0a52a",
        [
            {"job_id": "job_75b0bf3c5d5541f0a52a", "status": "queued", "progress": 10},
            {"job_id": "job_75b0bf3c5d5541f0a52a", "status": "running", "progress": 65},
            {
                "job_id": "job_75b0bf3c5d5541f0a52a",
                "status": "completed",
                "progress": 100,
                "result": {"ok": True},
            },
        ],
    )
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    exit_code = cli.main(
        [
            "poll",
            "--api-key",
            "sk_test",
            "--job",
            "job_75b0bf3c5d5541f0a52a",
            "--timeout",
            "90",
        ]
    )

    captured = capsys.readouterr()
    output = captured.out

    assert exit_code == 0
    assert "\rProgress: 10% (queued)" in output
    assert "\rProgress: 65% (running)" in output
    assert "\rProgress: 100% (completed)" in output
    assert "\n{\n" in output
    assert json.loads(output.split("\n", 1)[1]) == {
        "job_id": "job_75b0bf3c5d5541f0a52a",
        "status": "completed",
        "progress": 100,
        "result": {"ok": True},
    }
    assert fake_client.calls == [
        ("get_job", {"job_id": "job_75b0bf3c5d5541f0a52a"}),
        ("get_job", {"job_id": "job_75b0bf3c5d5541f0a52a"}),
        ("get_job", {"job_id": "job_75b0bf3c5d5541f0a52a"}),
    ]


def test_poll_command_updates_general_compile_file_for_compile_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_client = FakeClient(api_key="sk_test")
    fake_client.set_job_sequence(
        "job_compile_123",
        [
            {"job_id": "job_compile_123", "status": "running", "progress": 50, "type": "compile"},
            {
                "job_id": "job_compile_123",
                "status": "completed",
                "progress": 100,
                "type": "compile",
                "result": {"function_id": "fn_123"},
            },
        ],
    )
    _write_json(
        tmp_path / "temp_db" / "general" / "compile-output.json",
        {
            "job_id": "job_compile_123",
            "status": "queued",
        },
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    exit_code = cli.main(
        [
            "poll",
            "--api-key",
            "sk_test",
            "--job",
            "job_compile_123",
            "--timeout",
            "90",
        ]
    )

    captured = capsys.readouterr()
    saved = json.loads((tmp_path / "temp_db" / "general" / "compile-output.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert json.loads(captured.out.split("\n", 1)[1]) == {
        "job_id": "job_compile_123",
        "status": "completed",
        "progress": 100,
        "type": "compile",
        "result": {"function_id": "fn_123"},
    }
    assert saved == {
        "job_id": "job_compile_123",
        "status": "completed",
        "progress": 100,
        "type": "compile",
        "result": {"function_id": "fn_123"},
    }


def test_poll_command_updates_general_optimize_file_for_optimize_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_client = FakeClient(api_key="sk_test")
    fake_client.set_job_sequence(
        "job_optimize_123",
        [
            {"job_id": "job_optimize_123", "status": "running", "progress": 50, "type": "optimize"},
            {
                "job_id": "job_optimize_123",
                "status": "completed",
                "progress": 100,
                "type": "optimize",
                "result": {"optimized_prompt": "Better prompt"},
            },
        ],
    )
    _write_json(
        tmp_path / "temp_db" / "general" / "optimize-output.json",
        {
            "job_id": "job_optimize_123",
            "status": "queued",
        },
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    exit_code = cli.main(
        [
            "poll",
            "--api-key",
            "sk_test",
            "--job",
            "job_optimize_123",
            "--timeout",
            "90",
        ]
    )

    captured = capsys.readouterr()
    saved = json.loads((tmp_path / "temp_db" / "general" / "optimize-output.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert json.loads(captured.out.split("\n", 1)[1]) == {
        "job_id": "job_optimize_123",
        "status": "completed",
        "progress": 100,
        "type": "optimize",
        "result": {"optimized_prompt": "Better prompt"},
    }
    assert saved == {
        "job_id": "job_optimize_123",
        "status": "completed",
        "progress": 100,
        "type": "optimize",
        "result": {"optimized_prompt": "Better prompt"},
    }


def test_poll_command_updates_general_extract_file_for_extract_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_client = FakeClient(api_key="sk_test")
    fake_client.set_job_sequence(
        "job_extract_123",
        [
            {"job_id": "job_extract_123", "status": "running", "progress": 50, "type": "extract"},
            {
                "job_id": "job_extract_123",
                "status": "completed",
                "progress": 100,
                "type": "extract",
                "result": {"field": "value"},
            },
        ],
    )
    _write_json(
        tmp_path / "temp_db" / "general" / "extract-output.json",
        {
            "job_id": "job_extract_123",
            "status": "queued",
        },
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    exit_code = cli.main(
        [
            "poll",
            "--api-key",
            "sk_test",
            "--job",
            "job_extract_123",
            "--timeout",
            "90",
        ]
    )

    captured = capsys.readouterr()
    saved = json.loads((tmp_path / "temp_db" / "general" / "extract-output.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert json.loads(captured.out.split("\n", 1)[1]) == {
        "job_id": "job_extract_123",
        "status": "completed",
        "progress": 100,
        "type": "extract",
        "result": {"field": "value"},
    }
    assert saved == {
        "job_id": "job_extract_123",
        "status": "completed",
        "progress": 100,
        "type": "extract",
        "result": {"field": "value"},
    }


def test_config_command_uses_explicit_payload_file_and_saves_latest_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {
        "name": "Explicit config",
        "provider": "openrouter",
        "default_model": "google/gemini-2.5-flash-lite",
        "is_default": True,
        "byok": True,
    }
    payload_path = tmp_path / "config-input.json"
    _write_json(payload_path, payload)
    _write_json(tmp_path / "temp_db" / "general" / "config-output.json", {"config_id": "cfg_old"})
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("STABILIZER_PROVIDER_API_KEY", raising=False)
    reloaded = importlib.reload(cli)
    fake_client = FakeClient(api_key="sk_test")
    monkeypatch.setattr(reloaded, "_make_client", lambda api_key=None: fake_client)

    exit_code = reloaded.main(["config", "--api-key", "sk_test", "--payload-file", str(payload_path)])

    captured = capsys.readouterr()
    saved = json.loads((tmp_path / "temp_db" / "general" / "config-output.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert json.loads(captured.out) == {"config_id": "cfg_123", "status": "created"}
    assert fake_client.calls == [("create_llm_config", payload)]
    assert saved["config_id"] == "cfg_123"
    assert saved["status"] == "created"


def test_compile_command_does_not_update_general_without_poll(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {"name": "Example", "prompt": "Extract", "json_structure": {"field": "string"}}
    payload_path = tmp_path / "compile-input.json"
    _write_json(payload_path, payload)
    _write_json(tmp_path / "temp_db" / "general" / "compile-output.json", {"job_id": "job_old", "status": "queued"})
    fake_client = FakeClient(api_key="sk_test")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(["compile", "--api-key", "sk_test", "--payload-file", str(payload_path)])

    captured = capsys.readouterr()
    saved = json.loads((tmp_path / "temp_db" / "general" / "compile-output.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert json.loads(captured.out) == {"job_id": "job_compile", "status": "queued"}
    assert fake_client.calls == [("compile_function", payload)]
    assert saved == {"job_id": "job_old", "status": "queued"}


def test_optimize_command_does_not_update_general_without_poll(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {"prompt": "Extract", "json_structure": {"field": "string"}, "training_data": []}
    payload_path = tmp_path / "optimize-input.json"
    _write_json(payload_path, payload)
    _write_json(tmp_path / "temp_db" / "general" / "optimize-output.json", {"job_id": "job_old", "status": "queued"})
    fake_client = FakeClient(api_key="sk_test")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(["optimize", "--api-key", "sk_test", "--payload-file", str(payload_path)])

    captured = capsys.readouterr()
    saved = json.loads((tmp_path / "temp_db" / "general" / "optimize-output.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert json.loads(captured.out) == {"job_id": "job_optimize", "status": "queued"}
    assert fake_client.calls == [("optimize_prompt", payload)]
    assert saved == {"job_id": "job_old", "status": "queued"}


def test_poll_command_updates_optimize_output_after_separate_optimize_submission_without_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {"prompt": "Extract", "json_structure": {"field": "string"}, "training_data": []}
    payload_path = tmp_path / "optimize-input.json"
    _write_json(payload_path, payload)
    fake_client = FakeClient(api_key="sk_test")
    fake_client.set_job_sequence(
        "job_optimize",
        [
            {"job_id": "job_optimize", "status": "running", "progress": 50},
            {
                "job_id": "job_optimize",
                "status": "completed",
                "progress": 100,
                "result": {"optimized_prompt": "Better prompt"},
            },
        ],
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    submit_exit_code = cli.main(["optimize", "--api-key", "sk_test", "--payload-file", str(payload_path)])
    submit_output = capsys.readouterr()

    poll_exit_code = cli.main(["poll", "--api-key", "sk_test", "--job", "job_optimize", "--timeout", "90"])
    poll_output = capsys.readouterr()

    saved = json.loads((tmp_path / "temp_db" / "general" / "optimize-output.json").read_text(encoding="utf-8"))

    assert submit_exit_code == 0
    assert json.loads(submit_output.out) == {"job_id": "job_optimize", "status": "queued"}
    assert poll_exit_code == 0
    assert json.loads(poll_output.out.split("\n", 1)[1]) == {
        "job_id": "job_optimize",
        "status": "completed",
        "progress": 100,
        "result": {"optimized_prompt": "Better prompt"},
    }
    assert fake_client.calls == [
        ("optimize_prompt", payload),
        ("get_job", {"job_id": "job_optimize"}),
        ("get_job", {"job_id": "job_optimize"}),
    ]
    assert saved == {
        "job_id": "job_optimize",
        "status": "completed",
        "progress": 100,
        "result": {"optimized_prompt": "Better prompt"},
    }


def test_optimize_command_requires_payload_file(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "optimize-input.json").write_text(
        json.dumps({"prompt": "Extract", "json_structure": {"field": "string"}, "training_data": []}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(["optimize", "--api-key", "sk_test"])

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "--payload-file" in captured.err


def test_extract_command_does_not_update_general_without_poll(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {"function_id": "fn_replace_me", "source_text": "hello"}
    payload_path = tmp_path / "extract-input.json"
    _write_json(payload_path, payload)
    _write_json(tmp_path / "temp_db" / "general" / "extract-output.json", {"job_id": "job_old", "status": "queued"})
    fake_client = FakeClient(api_key="sk_test")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_make_client", lambda api_key=None: fake_client)

    exit_code = cli.main(["extract", "--api-key", "sk_test", "--payload-file", str(payload_path)])

    captured = capsys.readouterr()
    saved = json.loads((tmp_path / "temp_db" / "general" / "extract-output.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert json.loads(captured.out) == {"job_id": "job_extract", "status": "queued"}
    assert fake_client.calls == [("extract", payload)]
    assert saved == {"job_id": "job_old", "status": "queued"}


def test_extract_command_requires_payload_file(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "extract-input.json").write_text(
        json.dumps({"function_id": "fn_123", "source_text": "hello"}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(["extract", "--api-key", "sk_test"])

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "--payload-file" in captured.err

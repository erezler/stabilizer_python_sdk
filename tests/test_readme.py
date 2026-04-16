from __future__ import annotations

from pathlib import Path


def test_readme_documents_sdk_usage_paths() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    cli_example_lines = [
        line.strip()
        for line in readme.splitlines()
        if line.strip().startswith("py -m stabilizer_python_sdk")
    ]

    assert ".env.local" in readme
    assert "STABILIZER_API_KEY=YOUR_STABILIZER_API_KEY" in readme
    assert '$env:STABILIZER_API_KEY = "YOUR_STABILIZER_API_KEY"' in readme

    assert "py -m stabilizer_python_sdk.run_me" in readme
    assert "--state-file" in readme
    assert "--temp-db-dir" in readme
    assert "--compile-payload-file" in readme
    assert "--extract-payload-file" in readme
    assert cli_example_lines
    assert all("--api-key YOUR_STABILIZER_API_KEY" not in line for line in cli_example_lines)
    assert "Once `STABILIZER_API_KEY` is set via `.env.local` or your terminal session" in readme
    assert "you can omit `--api-key` from the command examples" in readme
    assert "`--api-key YOUR_STABILIZER_API_KEY` is optional" in readme
    assert "--new" in readme
    assert "--poll-interval" in readme
    assert "--poll-timeout" in readme

    assert "py -m stabilizer_python_sdk optimize" in readme
    assert "py -m stabilizer_python_sdk config" in readme
    assert "py -m stabilizer_python_sdk compile" in readme
    assert "py -m stabilizer_python_sdk extract" in readme
    assert "py -m stabilizer_python_sdk org" in readme
    assert "py -m stabilizer_python_sdk api-keys" in readme
    assert "py -m stabilizer_python_sdk configs" in readme
    assert "py -m stabilizer_python_sdk functions" in readme
    assert "py -m stabilizer_python_sdk function" in readme
    assert "py -m stabilizer_python_sdk job" in readme
    assert "py -m stabilizer_python_sdk poll" in readme
    assert "py -m stabilizer_python_sdk usage" in readme
    assert "py -m stabilizer_python_sdk evaluate-variance" in readme
    assert "py -m stabilizer_python_sdk evaluate-gt" in readme
    assert "py -m stabilizer_python_sdk state latest" in readme
    assert "temp_db\\general" in readme
    assert "config-input.json" in readme
    assert "compile-output.json" in readme

    assert "run only the commands you need" in readme.lower()
    assert "StabilizerAdminClient" not in readme
    assert "/v1/admin/" not in readme
    assert "admin api" not in readme.lower()

from __future__ import annotations

from pathlib import Path


def test_readme_documents_sdk_usage_paths() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert ".env.local" in readme
    assert "STABILIZER_API_KEY=YOUR_STABILIZER_API_KEY" in readme
    assert '$env:STABILIZER_API_KEY = "YOUR_STABILIZER_API_KEY"' in readme

    assert "py -m stabilizer_python_sdk.run_me" in readme
    assert "--state-file" in readme
    assert "--temp-db-dir" in readme
    assert "--compile-payload-file" in readme
    assert "--extract-payload-file" in readme
    assert "--api-key" in readme
    assert "--new" in readme
    assert "--poll-interval" in readme
    assert "--poll-timeout" in readme

    assert "py -m stabilizer_python_sdk optimize" in readme
    assert "py -m stabilizer_python_sdk config" in readme
    assert "py -m stabilizer_python_sdk compile" in readme
    assert "py -m stabilizer_python_sdk extract" in readme
    assert "py -m stabilizer_python_sdk poll" in readme
    assert "py -m stabilizer_python_sdk state list" in readme
    assert "py -m stabilizer_python_sdk state latest" in readme
    assert "config.json" in readme

    assert "run only the commands you need" in readme.lower()

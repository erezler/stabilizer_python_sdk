from __future__ import annotations

import runpy

import pytest


def test_package_exposes_version() -> None:
    import stabilizer_python_sdk

    assert stabilizer_python_sdk.__version__ == "0.1.0"


def test_module_entrypoint_prints_project_name(capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_module("stabilizer_python_sdk", run_name="__main__")

    captured = capsys.readouterr()

    assert captured.out.strip() == "stabilizer_python_sdk"

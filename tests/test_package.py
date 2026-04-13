from __future__ import annotations

import stabilizer_python_sdk.__main__ as cli


def test_package_exposes_version() -> None:
    import stabilizer_python_sdk

    assert stabilizer_python_sdk.__version__ == "0.1.0"


def test_module_entrypoint_main_returns_success_for_help(capsys) -> None:
    exit_code = cli.main([])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "usage:" in captured.out

from __future__ import annotations

import stabilizer_python_sdk
import stabilizer_python_sdk.__main__ as cli


def test_package_exposes_version() -> None:
    assert stabilizer_python_sdk.__version__ == "0.2.0"


def test_module_entrypoint_main_returns_success_for_help(capsys) -> None:
    exit_code = cli.main([])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "usage:" in captured.out


def test_package_does_not_expose_admin_client() -> None:
    assert not hasattr(stabilizer_python_sdk, "StabilizerAdminClient")

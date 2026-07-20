from __future__ import annotations

import stabilizer_python_sdk
import stabilizer_python_sdk.__main__ as cli


def test_package_exposes_version() -> None:
    assert stabilizer_python_sdk.__version__ == "0.6.0"


def test_module_entrypoint_main_returns_success_for_help(capsys) -> None:
    exit_code = cli.main([])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "usage:" in captured.out


def test_package_exposes_admin_client() -> None:
    assert hasattr(stabilizer_python_sdk, "StabilizerAdminClient")
    assert "StabilizerAdminClient" in stabilizer_python_sdk.__all__


def test_admin_client_is_not_reachable_from_the_cli() -> None:
    """Admin keys stay out of the CLI surface; admin ops are library-only."""
    assert "StabilizerAdminClient" not in dir(cli)

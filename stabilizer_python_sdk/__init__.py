from __future__ import annotations

from pathlib import Path

_SRC_PACKAGE_DIR = (
    Path(__file__).resolve().parent.parent / "src" / "stabilizer_python_sdk"
)

if str(_SRC_PACKAGE_DIR) not in __path__:
    __path__.append(str(_SRC_PACKAGE_DIR))

_SRC_INIT = _SRC_PACKAGE_DIR / "__init__.py"
exec(_SRC_INIT.read_text(encoding="utf-8"), globals())

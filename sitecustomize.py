from __future__ import annotations

import sys
from pathlib import Path


def _add_src_dir_to_path() -> None:
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if not src_dir.is_dir():
        return

    normalized_src = str(src_dir)
    if normalized_src not in sys.path:
        sys.path.insert(0, normalized_src)


_add_src_dir_to_path()

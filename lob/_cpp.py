from __future__ import annotations

import sys
from pathlib import Path

try:
    import lob_cpp  # type: ignore
except ModuleNotFoundError:
    root = Path(__file__).resolve().parents[1]
    build = root / "cpp" / "build"
    if build.exists():
        sys.path.insert(0, str(build))
    import lob_cpp  # type: ignore

__all__ = ["lob_cpp"]
